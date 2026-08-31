"""
The chat agent.

Structural change from the previous version: the expensive, seller-independent
parts (tool argument schemas, the system prompt) moved out to agent/schemas.py
and agent/prompts.py so they're built once at import instead of once per request,
and callers now reuse one OptimizedChatbot per conversation via
services/session_store.py rather than constructing a fresh agent - 20 Pydantic
model classes, 20 tools, a bound LLM and an executor - for every message.

process_message stays synchronous and blocking. Callers must run it off the event
loop (see routes/chat.py and routes/whatsapp_routes.py); it makes network calls
that would otherwise stall every other request in the process.
"""
import re
import threading
import time
from typing import Any, Dict, List, Optional, Union

from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.tools import StructuredTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.prompts import get_unified_system_prompt
from agent.schemas import (
    AddItemToOrderInput,
    CancelOrderInput,
    CheckStockInput,
    EmptyInput,
    EscalateToHumanInput,
    GetOrderDetailsInput,
    GetProductInfoInput,
    PlaceOrderInput,
    ProductImageSearchInput,
    RemoveItemFromOrderInput,
    ReplaceOrderItemsInput,
    SaveUserInput,
    TrackOrderInput,
    UpdateItemQuantityInput,
    UpdateUserInfoInput,
    VerifyPaymentProofInput,
)
from config.settings import settings
from repositories.tools import (
    add_item_to_order,
    cancel_order,
    check_product_stock,
    check_user_exists,
    create_tmp_user_id,
    find_similar_products_by_image,
    get_all_orders_for_customer,
    get_all_products,
    get_order_details,
    get_pending_orders,
    get_product_info,
    get_user_info,
    log_query,
    place_order_detailed,
    remove_item_from_order,
    replace_order_items,
    save_user,
    track_order_detailed,
    update_item_quantity_in_order,
    update_user_info,
    verify_and_save_payment_proof_detailed,
)
from services import cost_optimizer, outbound_formatter
from services.usage_tracker import TokenUsageCallback, usage_tracker
from templates.message_templates import MessageTemplates
from utils import background
from utils.cache import get_cache
from utils.logger import get_logger
from utils.metrics import metrics

logger = get_logger(__name__)

# Re-exported for callers that import these from here (routes/chat.py does).
__all__ = [
    "OptimizedChatbot",
    "create_optimized_chatbot",
    "create_multi_agent_system",
    "fast_intent_detection",
    "llm",
    "get_llm",
    "log_query",
    "check_user_exists",
    "create_tmp_user_id",
]


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------
def format_product_info_response(product_data):
    """Format product info using beautiful template"""
    return MessageTemplates.product_details(product_data)


def format_product_list_response(products_data):
    """Format product list using beautiful template"""
    return MessageTemplates.product_list(products_data)


def format_order_details_response(order_data):
    """Format order details using beautiful template"""
    return MessageTemplates.order_details(order_data)


def format_tracking_response(tracking_data):
    """Format tracking info using beautiful template"""
    return MessageTemplates.tracking_status(tracking_data)


def format_customer_info_response(customer_data):
    """Format customer info using beautiful template"""
    return MessageTemplates.customer_info(customer_data)


def format_payment_confirmation_response(payment_data):
    """Format payment confirmation using beautiful template"""
    return MessageTemplates.payment_confirmation(payment_data)


def format_error_response(error_type, details=""):
    """Format error message using beautiful template"""
    return MessageTemplates.error_message(error_type, details)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
_llm_cache: Dict[str, Any] = {}
_llm_lock = threading.Lock()


def _build_llm(model: Optional[str] = None):
    """Build the chat model for the configured provider.

    Previously an unrecognised AI_PROVIDER left the module-level `llm` name
    undefined, so the app imported fine and then died with NameError on the first
    customer message. Now it fails at startup with a message that says why.
    """
    provider = settings.ai_provider
    model_name = model or settings.chat_model

    if not settings.api_key:
        logger.warning("API_KEY is not set - LLM calls will fail authentication")

    common = dict(
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )

    if provider == "DEEPSEEK":
        from langchain_deepseek.chat_models import ChatDeepSeek

        return ChatDeepSeek(
            model=model_name,
            api_key=settings.api_key,
            base_url=settings.api_base,
            **common,
        )

    if provider == "GPT":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=settings.api_key,
            base_url=settings.api_base,
            **common,
        )

    raise ValueError(
        f"Unsupported AI_PROVIDER={provider!r}. Set AI_PROVIDER to 'GPT' (any "
        f"OpenAI-compatible endpoint, including OpenRouter) or 'DEEPSEEK'."
    )


def get_llm(model: Optional[str] = None):
    """A chat model instance, one per model name.

    Cached because cost routing can switch models between turns and rebuilding
    the client each time would throw away its HTTP connection pool.
    """
    key = model or settings.chat_model
    with _llm_lock:
        instance = _llm_cache.get(key)
        if instance is None:
            instance = _build_llm(key)
            _llm_cache[key] = instance
            logger.info("Initialised chat model %s", key)
        return instance


# Default model instance. Kept as a module-level name because main.py warms it up.
llm = get_llm()


# ---------------------------------------------------------------------------
# RAG examples
# ---------------------------------------------------------------------------
# Was @lru_cache(maxsize=100): no expiry, so a changed knowledge base was never
# picked up, and every distinct customer phrasing occupied a slot forever.
_rag_cache = get_cache("rag_examples", maxsize=256, ttl=900)


def get_cached_rag_examples(
    user_input: str, seller_id: str, k: int = 2, threshold: float = 3
) -> str:
    """Cached RAG examples with reduced k for speed"""

    def _lookup() -> str:
        try:
            from vector_store.vector_store import fast_vector_store as vector_store

            results = vector_store.similarity_search(user_input, k=k, threshold=threshold)
            examples = []
            for result in results:
                if hasattr(result, "metadata"):
                    result_intent = result.metadata.get("intent")
                    examples.append(f"{result.page_content[:200]}... (Intent: {result_intent})")
            return "\n".join(examples[:k]) if examples else ""
        except Exception as e:
            logger.warning("[RAG Cache] Error: %s", e)
            return ""

    key = f"{seller_id}|{k}|{threshold}|{user_input.strip().lower()}"
    return _rag_cache.get_or_set(key, _lookup)


def get_cached_intents(seller_id: str) -> str:
    """The intent vocabulary. Constant - kept as a function for compatibility."""
    return "product_info, order_tracking, place_order, user_management, general_inquiry"


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
# Rule-based, so it costs no LLM call. Intent feeds analytics and the fallback
# copy, not tool choice, so being approximate is fine - but the old version
# listed 'order' under three different intents and checked order_tracking first,
# so "I want to order a laptop" was always classified as order_tracking.
# Patterns are now ordered most-specific-first and compiled once.
_INTENT_PATTERNS = (
    (
        "order_tracking",
        re.compile(
            r"\b(track|tracking|where is my order|order status|delivery status|shipped|dispatched)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "place_order",
        re.compile(
            r"\b(buy|purchase|checkout|cart|place an? order|want to order|i'?ll take|order now)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "user_management",
        re.compile(
            r"\b(my profile|my account|update my|change my|my details|my address|my email)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "product_info",
        re.compile(
            r"\b(product|products|price|cost|how much|available|availability|in stock|stock|catalogue|catalog)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "order_tracking",
        re.compile(r"\b(order|orders)\b", re.IGNORECASE),
    ),
)


def fast_intent_detection(user_input: str) -> str:
    """Fast rule-based intent detection for multiple languages"""
    if not user_input:
        return "general_inquiry"
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(user_input):
            return intent
    return "general_inquiry"


# Emoji that mark a tool response as already template-formatted. Checked with a
# compiled alternation rather than eleven startswith calls per tool result.
_TEMPLATE_PREFIX = re.compile(
    r"^(🛍️|🚚|📋|🛒|👤|💰|🔍|📦|⚠️|🔧|💳)"
)


def _is_template_response(text: str) -> bool:
    return bool(_TEMPLATE_PREFIX.match(text or ""))


class OptimizedChatbot:
    """Optimized single-agent chatbot for faster responses.

    One instance per conversation, reused across messages. Building an instance
    binds 20 tools and an executor, so don't create one per request - go through
    services.session_store.session_store.
    """

    def __init__(self, seller_id: str, user_id: str):
        self.seller_id = str(seller_id)
        self.user_id = str(user_id)
        self.chat_history: List[Dict[str, str]] = []
        self.tools = self._create_tools()
        self._prompt = self._build_prompt()
        self._executors: Dict[str, AgentExecutor] = {}
        self.default_model = settings.chat_model
        # Kept as .agent for any caller that reaches for it directly.
        self.agent = self._executor_for(self.default_model)
        self.last_usage = None
        self.last_model = self.default_model

        # Per-turn record of what the tools did. Two separate things are tracked:
        # `result` is what the customer sees, `data` is the structured payload we
        # mine for entities and image URLs. Conflating them was why entity
        # extraction always came back empty for products.
        self.last_tool_results: List[Dict[str, Any]] = []
        self.escalated = False

    # -- history -------------------------------------------------------------
    def load_history(self, history: List[Dict[str, str]]) -> None:
        """Seed the conversation, e.g. from the database after a restart."""
        self.chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in (history or [])
            if m.get("role") in ("user", "assistant") and m.get("content")
        ][-settings.max_chat_history :]

    def _record(self, tool_name: str, result: Any, data: Any = None) -> None:
        self.last_tool_results.append(
            {"tool_name": tool_name, "result": result, "data": data}
        )

    def _create_tools(self):
        """Create tools bound to this conversation's seller and customer.

        The argument schemas come from agent/schemas.py - only the closures below
        are per-instance. seller_id and user_id are captured here and never taken
        from the model, so the model cannot act on another customer's data.
        """

        def get_product_info_wrapper(product_name: str) -> str:
            result = get_product_info(seller_id=self.seller_id, product_name=product_name)
            if isinstance(result, dict) and "error" not in result:
                formatted = format_product_info_response(result)
                self._record("get_product_info", formatted, data=result)
                return formatted
            self._record("get_product_info", None, data=result)
            return format_error_response(
                "not_found", f"Product '{product_name}' not found in our catalog"
            )

        def track_order_wrapper(order_id: str) -> str:
            result = track_order_detailed(order_id=order_id)
            if not result.get("found"):
                self._record("track_order", None, data=None)
                return format_error_response("not_found", f"Order #{order_id} not found")

            summary = (
                f"Order ID: {result['order_id']}, Status: {result['status']}, "
                f"Created: {result['created_at']}"
            )
            tracking_data = {
                "order_id": result["order_id"],
                "status": result["status"],
                "details": summary,
                "total_amount": result.get("total_amount"),
            }
            formatted = format_tracking_response(tracking_data)
            self._record("track_order", formatted, data=tracking_data)
            return formatted

        def place_order_wrapper(items: List[dict]) -> str:
            if not items or not isinstance(items, list):
                logger.error("[place_order_wrapper] Invalid items parameter: %s", items)
                return format_error_response(
                    "invalid_input",
                    "Invalid items parameter. Expected list of dictionaries with product_id and quantity.",
                )

            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    logger.error("[place_order_wrapper] Item %d is not a dictionary: %s", i, item)
                    return format_error_response(
                        "invalid_input",
                        f"Item {i} must be a dictionary with product_id and quantity keys.",
                    )
                if "product_id" not in item or "quantity" not in item:
                    logger.error("[place_order_wrapper] Item %d missing required keys: %s", i, item)
                    return format_error_response(
                        "invalid_input",
                        f"Item {i} must have both 'product_id' and 'quantity' keys.",
                    )

            result = place_order_detailed(
                seller_id=self.seller_id, user_id=self.user_id, items=items
            )
            message = result.get("message", "")
            lowered = message.lower()

            if not result.get("success"):
                if "insufficient stock" in lowered:
                    return format_error_response("out_of_stock", message)
                return format_error_response("system_error", message)

            order_data = {
                "order_id": result.get("order_id"),
                "total_amount": result.get("total_amount"),
                "items": items,
                "status": "pending",
                "result": message,
            }
            formatted = format_order_details_response(order_data)
            self._record("place_order", formatted, data=order_data)
            return formatted

        def save_user_wrapper(name: str, email: str, address: str, number: str) -> str:
            return save_user(
                user_id=self.user_id, name=name, email=email, address=address, number=number
            )

        def get_user_info_wrapper() -> str:
            result = get_user_info(user_id=self.user_id)
            if "not found" in str(result).lower() or "does not exist" in str(result).lower():
                return format_error_response("not_found", "User information not found")
            return format_customer_info_response(result)

        def check_user_exists_wrapper() -> bool:
            return check_user_exists(user_id=self.user_id)

        def get_all_products_wrapper() -> str:
            result = get_all_products(seller_id=self.seller_id)
            if not result or "no products" in str(result).lower():
                return format_error_response("not_found", "No products available at the moment")
            formatted = format_product_list_response(result)
            self._record("get_all_products", formatted, data=result)
            return formatted

        def update_user_info_wrapper(
            name: str = "", email: str = "", address: str = "", number: str = ""
        ) -> str:
            return update_user_info(
                user_id=self.user_id,
                name=name or None,
                email=email or None,
                address=address or None,
                number=number or None,
            )

        def add_item_to_order_wrapper(order_id: str, product_identifier: str, quantity: int) -> str:
            return add_item_to_order(
                customer_id=self.user_id,
                order_id=order_id,
                product_identifier=product_identifier,
                quantity=quantity,
            )

        def remove_item_from_order_wrapper(order_id: str, product_identifier: str) -> str:
            return remove_item_from_order(
                customer_id=self.user_id,
                order_id=order_id,
                product_identifier=product_identifier,
            )

        def update_item_quantity_in_order_wrapper(
            order_id: str, product_identifier: str, new_quantity: int
        ) -> str:
            return update_item_quantity_in_order(
                customer_id=self.user_id,
                order_id=order_id,
                product_identifier=product_identifier,
                new_quantity=new_quantity,
            )

        def replace_order_items_wrapper(order_id: str, new_items: List[dict]) -> str:
            return replace_order_items(
                customer_id=self.user_id, order_id=order_id, new_items=new_items
            )

        def get_all_orders_for_customer_wrapper() -> list:
            return get_all_orders_for_customer(customer_id=self.user_id)

        def get_pending_orders_wrapper() -> list:
            return get_pending_orders(customer_id=self.user_id)

        def get_order_details_wrapper(order_id: int) -> str:
            result = get_order_details(order_id=order_id)
            if not result or result.get("error"):
                return format_error_response("not_found", f"Order #{order_id} not found")
            formatted = format_order_details_response(result)
            self._record("get_order_details", formatted, data=result)
            return formatted

        def check_product_stock_wrapper(product_id: int, quantity: int) -> dict:
            result = check_product_stock(product_id=product_id, quantity=quantity)
            self._record("check_product_stock", None, data={"product_id": product_id})
            return result

        def verify_payment_proof_wrapper(order_id: str, payment_proof_file: str) -> str:
            # customer_id is bound from the session, never from the LLM, so a
            # customer cannot attach a receipt to someone else's order.
            result = verify_and_save_payment_proof_detailed(
                order_id=order_id,
                file_path=payment_proof_file,
                customer_id=self.user_id,
            )
            message = result.get("message") or "I couldn't process that receipt."
            # Only hand the templates structured data when the check actually ran;
            # otherwise the message is an error string and should pass through.
            self._record(
                "verify_and_save_payment_proof",
                message,
                data=result if result.get("verification") else None,
            )
            return message

        def find_similar_products_by_image_wrapper(image_file: str) -> str:
            return find_similar_products_by_image(
                file_path=image_file, seller_id=self.seller_id
            )

        def cancel_order_wrapper(order_id: str, reason: str = "") -> str:
            return cancel_order(customer_id=self.user_id, order_id=order_id, reason=reason)

        def escalate_to_human_wrapper(reason: str) -> str:
            from services import handoff

            handoff.request(
                seller_id=self.seller_id,
                user_id=self.user_id,
                reason=reason,
                last_message=self.chat_history[-1]["content"] if self.chat_history else None,
            )
            self.escalated = True
            self._record("escalate_to_human", None, data={"reason": reason})
            return (
                "I've passed this on to our team - someone will get back to you shortly. "
                "Is there anything else I can help with in the meantime?"
            )

        return [
            StructuredTool(
                name="get_product_info",
                func=get_product_info_wrapper,
                description="Get detailed product information including price, description, and images by product name or ID. Example: get_product_info(product_name='laptop') or get_product_info(product_name='iPhone 15')",
                args_schema=GetProductInfoInput,
            ),
            StructuredTool(
                name="track_order",
                func=track_order_wrapper,
                description="Track order status and delivery information by order ID. Returns current status, estimated delivery, tracking details. Example: track_order(order_id='12345') or track_order(order_id='ORD001')",
                args_schema=TrackOrderInput,
            ),
            StructuredTool(
                name="place_order",
                func=place_order_wrapper,
                description="Place a new order with specified items and quantities. Each item must have 'product_id' and 'quantity'. Example: place_order(items=[{'product_id': 1, 'quantity': 2}, {'product_id': 'laptop', 'quantity': 1}]). NEVER use empty dict {}!",
                args_schema=PlaceOrderInput,
            ),
            StructuredTool(
                name="save_user",
                func=save_user_wrapper,
                description="Create a new user account with personal details (name, email, address, phone). Required for first-time customers. Example: save_user(name='John Smith', email='john@email.com', address='123 Main St', number='+94771234567')",
                args_schema=SaveUserInput,
            ),
            StructuredTool(
                name="get_user_info",
                func=get_user_info_wrapper,
                description="Retrieve current user's profile information (name, email, address, phone). No parameters needed - uses current user context. Example: get_user_info()",
                args_schema=EmptyInput,
            ),
            StructuredTool(
                name="check_user_exists",
                func=check_user_exists_wrapper,
                description="Check if current user exists in the system. Returns True/False. Essential before placing orders. No parameters needed. Example: check_user_exists()",
                args_schema=EmptyInput,
            ),
            StructuredTool(
                name="update_user_info",
                func=update_user_info_wrapper,
                description="Update specific user profile fields. Only provide fields to change, leave others empty. Example: update_user_info(name='New Name', email='') to only update name",
                args_schema=UpdateUserInfoInput,
            ),
            StructuredTool(
                name="get_all_products",
                func=get_all_products_wrapper,
                description="Get complete list of all available products from the seller's catalog. No parameters needed. Example: get_all_products()",
                args_schema=EmptyInput,
            ),
            StructuredTool(
                name="add_item_to_order",
                func=add_item_to_order_wrapper,
                description="Add a new item to existing pending order or increase quantity if item exists. Example: add_item_to_order(order_id='12345', product_identifier='laptop', quantity=1)",
                args_schema=AddItemToOrderInput,
            ),
            StructuredTool(
                name="remove_item_from_order",
                func=remove_item_from_order_wrapper,
                description="Completely remove a specific item from existing pending order. Example: remove_item_from_order(order_id='12345', product_identifier='laptop')",
                args_schema=RemoveItemFromOrderInput,
            ),
            StructuredTool(
                name="update_item_quantity_in_order",
                func=update_item_quantity_in_order_wrapper,
                description="Change quantity of existing item in pending order. Example: update_item_quantity_in_order(order_id='12345', product_identifier='laptop', new_quantity=3)",
                args_schema=UpdateItemQuantityInput,
            ),
            StructuredTool(
                name="replace_order_items",
                func=replace_order_items_wrapper,
                description="Replace ALL items in pending order with completely new item list. Example: replace_order_items(order_id='12345', new_items=[OrderItemInput(product_id=1, quantity=2)])",
                args_schema=ReplaceOrderItemsInput,
            ),
            StructuredTool(
                name="get_all_orders_for_customer",
                description="Retrieve complete order history for current customer including all statuses (pending, confirmed, delivered). No parameters needed. Example: get_all_orders_for_customer()",
                func=get_all_orders_for_customer_wrapper,
                args_schema=EmptyInput,
            ),
            StructuredTool(
                name="get_pending_orders",
                description="Get only pending/unpaid orders for current customer. Useful for order editing. No parameters needed. Example: get_pending_orders()",
                func=get_pending_orders_wrapper,
                args_schema=EmptyInput,
            ),
            StructuredTool(
                name="get_order_details",
                description="Get detailed information about specific order including items, status, total cost. Example: get_order_details(order_id='12345') or get_order_details(order_id=67890)",
                func=get_order_details_wrapper,
                args_schema=GetOrderDetailsInput,
            ),
            StructuredTool(
                name="check_product_stock",
                description="Verify if sufficient stock available before placing/editing orders. Example: check_product_stock(product_id=1, quantity=5) or check_product_stock(product_id='laptop', quantity=2)",
                func=check_product_stock_wrapper,
                args_schema=CheckStockInput,
            ),
            StructuredTool(
                name="verify_and_save_payment_proof",
                func=verify_payment_proof_wrapper,
                description=(
                    "Use this when a customer sends a payment receipt or bank transfer slip image. "
                    "It reads the receipt, checks the amount against the order total, saves it, and "
                    "flags the order for review if anything is wrong. Returns the message to give the "
                    "customer - return it as-is. "
                    "Example: verify_and_save_payment_proof(order_id='7', payment_proof_file='./downloads/whatsapp_image_123.jpg')"
                ),
                args_schema=VerifyPaymentProofInput,
            ),
            StructuredTool(
                name="find_similar_products_by_image",
                func=find_similar_products_by_image_wrapper,
                description=(
                    "Use this when a customer sends a photo of a product and wants to know if we have "
                    "it or something similar. Looks at the image and returns matching products from the "
                    "catalogue. Return the result as-is. "
                    "Example: find_similar_products_by_image(image_file='./downloads/whatsapp_image_123.jpg')"
                ),
                args_schema=ProductImageSearchInput,
            ),
            StructuredTool(
                name="cancel_order",
                func=cancel_order_wrapper,
                description="Cancel a pending order and restore product stock. Only pending orders can be cancelled. Example: cancel_order(order_id='12345', reason='Changed mind')",
                args_schema=CancelOrderInput,
            ),
            StructuredTool(
                name="escalate_to_human",
                func=escalate_to_human_wrapper,
                description=(
                    "Hand the conversation to a member of shop staff. Use for refunds, complaints, "
                    "angry customers, changes to paid orders, an explicit request for a person, or "
                    "anything your other tools cannot do. The bot stops auto-replying to this "
                    "customer until staff clear it. Example: escalate_to_human(reason='Customer wants "
                    "a refund on order 12')"
                ),
                args_schema=EscalateToHumanInput,
            ),
        ]

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    get_unified_system_prompt(self.seller_id) + "\n\nContext Examples: {examples}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def _create_agent(self, model: Optional[str] = None) -> AgentExecutor:
        """Create optimized single agent with unified language support"""
        llm_with_tools = get_llm(model).bind_tools(self.tools)
        agent = create_openai_tools_agent(llm_with_tools, self.tools, self._prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=settings.agent_max_iterations,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )

    def _executor_for(self, model: str) -> AgentExecutor:
        """Executor for a given model, built once and reused.

        Cost routing can pick a different model per turn. Re-binding the tools and
        rebuilding the executor every time would undo the point of reusing the
        session, so each model gets one executor for the session's lifetime.
        """
        executor = self._executors.get(model)
        if executor is None:
            executor = self._create_agent(model)
            self._executors[model] = executor
        return executor

    # -- tool result accessors ----------------------------------------------
    def get_tool_results(self) -> List[Dict[str, Any]]:
        """Get the tool results from the last conversation turn"""
        return list(self.last_tool_results)

    def clear_tool_results(self):
        """Clear tool results"""
        self.last_tool_results = []

    def extract_id_from_str(self, result_data, id_type) -> str:
        """Extract ID from result data based on type"""
        if isinstance(result_data, dict):
            # Tolerate both "Product ID" and "product_id" spellings.
            for key in (id_type, id_type.lower().replace(" ", "_")):
                if key in result_data:
                    return str(result_data[key])

        match = re.search(rf"{re.escape(id_type)}:\s*(\d+)", str(result_data))
        return match.group(1) if match else ""

    def get_img_urls(self) -> List[str]:
        """Get image URLs from the last tool results"""
        urls: List[str] = []
        for item in self.last_tool_results:
            if item.get("tool_name") != "get_product_info":
                continue
            data = item.get("data")
            if isinstance(data, dict):
                images = data.get("images") or []
                urls.extend(str(u) for u in images if u)
        # De-duplicate while preserving order, and cap it - WhatsApp charges per
        # message and a customer doesn't want ten photos.
        seen = set()
        unique = [u for u in urls if not (u in seen or seen.add(u))]
        return unique[:3]

    def extract_entities(self) -> Dict[str, Any]:
        """Extract entities from the last tool results"""
        entities: Dict[str, Any] = {}
        for item in self.last_tool_results:
            tool_name = item.get("tool_name", "")
            data = item.get("data")

            if tool_name == "get_product_info":
                product_id = self.extract_id_from_str(data, "product_id")
                if product_id:
                    entities["product_id"] = product_id
            elif tool_name in ("track_order", "get_order_details"):
                if isinstance(data, dict) and data.get("order_id"):
                    entities["order_id"] = str(data["order_id"])
            elif tool_name == "check_product_stock":
                if isinstance(data, dict) and data.get("product_id"):
                    entities["product_id"] = str(data["product_id"])
            elif tool_name == "escalate_to_human":
                entities["escalated"] = True
                if isinstance(data, dict):
                    entities["escalation_reason"] = data.get("reason")
        return entities

    def log_query(
        self,
        query: str,
        intent: str,
        response: str,
        entities: Union[str, Dict[str, Any], List] = "",
        response_time: int = 0,
    ):
        """Log query synchronously"""
        try:
            log_query(
                query=query,
                intent=intent,
                entities=entities,
                response=response,
                seller_id=self.seller_id,
                user_id=self.user_id,
                response_time=response_time,
            )
        except Exception as e:
            logger.error("[Optimized] logging error: %s", e)

    # -- main entry point ----------------------------------------------------
    def process_message(self, message: str, external_chat_history: List[Dict] = None) -> str:
        """Process message with language detection and optimizations.

        Blocking: makes LLM and database calls. Callers on an event loop must run
        it in a thread (fastapi.concurrency.run_in_threadpool).
        """
        start_time = time.time()
        self.clear_tool_results()
        self.escalated = False
        self.last_usage = None

        try:
            if external_chat_history:
                self.load_history(external_chat_history)

            history_length = len(self.chat_history)

            # Cheapest path first: a bare greeting or "thanks" is answered from a
            # template for zero tokens. Only exact matches take this path.
            shortcut = cost_optimizer.try_shortcut_reply(
                message, history_length, self.seller_id
            )
            if shortcut is not None:
                return self._finish_turn(
                    message,
                    shortcut,
                    "general_inquiry",
                    start_time,
                    history_length,
                    from_cache=True,
                )

            cached = cost_optimizer.cached_response(self.seller_id, message, history_length)
            if cached is not None:
                logger.debug("[Cost] Reusing cached reply for an identical opening question")
                return self._finish_turn(
                    message,
                    cached,
                    fast_intent_detection(message),
                    start_time,
                    history_length,
                    from_cache=True,
                )

            self.chat_history.append({"role": "user", "content": message})

            detected_language = "unknown"
            if settings.language_detection_enabled:
                from agent.language_agent import detect_language

                detected_language = detect_language(message)
                logger.info("[Language] Detected language: %s", detected_language)

            intent = fast_intent_detection(message)
            logger.debug("[Optimized] Detected intent: %s", intent)

            examples = ""
            if settings.rag_enabled:
                examples = get_cached_rag_examples(message, self.seller_id, k=3)
                logger.debug("[Optimized] Retrieved RAG examples for intent %s", intent)

            # Real message objects, so LangChain doesn't have to coerce tuples on
            # every turn. Exclude the message we just appended - it goes in via
            # the "{input}" slot, and passing it twice made the model see the
            # customer's question duplicated.
            formatted_history = [
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in self.chat_history[:-1][-settings.max_chat_history :]
            ]

            # Which model this turn is worth. 'fixed' (the default) always
            # returns CHAT_MODEL, so this is a no-op unless COST_STRATEGY is set.
            model = cost_optimizer.choose_model(
                message=message,
                intent=intent,
                history_length=history_length,
                seller_id=self.seller_id,
            )
            self.last_model = model

            response = self._invoke_agent(
                message, examples, intent, formatted_history, model
            )

            logger.info(
                "[Optimized] Language: %s, model: %s",
                detected_language,
                model,
            )
            return self._finish_turn(
                message, response, intent, start_time, history_length
            )

        except Exception as e:
            metrics.incr("agent.turn.fatal")
            logger.error("[Optimized] Error: %s", e, exc_info=True)
            return self._fallback_message(message, technical=True)

    def _finish_turn(
        self,
        message: str,
        response: str,
        intent: str,
        start_time: float,
        history_length: int = 0,
        from_cache: bool = False,
    ) -> str:
        """Apply the outbound templates, then record the turn.

        Everything that must happen whether the reply came from the agent, the
        response cache, or a shortcut lives here so the three paths can't drift.
        """
        used_tools = bool(self.last_tool_results)

        # The seller's templates get the last word on what actually goes out.
        # Skip it for cached/shortcut replies - those are already template output.
        if not from_cache:
            try:
                response = outbound_formatter.format_reply(
                    response=response,
                    tool_results=self.last_tool_results,
                    seller_id=self.seller_id,
                    escalated=self.escalated,
                )
            except Exception as e:
                # A broken template must not cost the customer their answer.
                logger.error("Outbound template formatting failed: %s", e, exc_info=True)

        # Keep the turn in history. A shortcut reply skipped the append above.
        if not self.chat_history or self.chat_history[-1]["role"] != "user":
            self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response})
        if len(self.chat_history) > settings.max_chat_history:
            self.chat_history = self.chat_history[-settings.max_chat_history :]

        total_time = time.time() - start_time
        metrics.observe("agent.turn", total_time)
        metrics.incr(f"agent.intent.{intent}")
        logger.info("[Optimized] Total processing time: %.2fs", total_time)

        if self.last_usage is not None and settings.track_token_usage:
            usage_tracker.record(self.seller_id, self.user_id, self.last_usage)

        cost_optimizer.remember_response(
            seller_id=self.seller_id,
            message=message,
            response=response,
            # The pre-turn length, passed in rather than derived from the list
            # after two appends and a possible trim.
            history_length=history_length,
            used_tools=used_tools,
        )

        entities = self.extract_entities()
        background.submit(
            self.log_query,
            message,
            intent,
            response,
            entities,
            int(total_time * 1000),
            task_name="log_query",
        )

        if settings.persist_conversations:
            from repositories import conversations

            background.submit(
                conversations.append_turn,
                self.seller_id,
                self.user_id,
                message,
                response,
                task_name="persist_turn",
            )

        return response

    def _invoke_agent(
        self,
        message: str,
        examples: str,
        intent: str,
        formatted_history: List,
        model: Optional[str] = None,
    ) -> str:
        """Run the executor and turn whatever comes back into a reply."""
        model = model or self.default_model
        # One callback per turn. A turn makes one LLM call per tool round trip and
        # the executor only returns the last message, so usage has to be collected
        # as the calls happen.
        usage_callback = TokenUsageCallback(model_hint=model)

        try:
            result = self._executor_for(model).invoke(
                {
                    "input": message,
                    "examples": examples,
                    "intent": intent,
                    "chat_history": formatted_history,
                },
                config={"callbacks": [usage_callback]},
            )
            response = result.get("output") or "I couldn't process your request."

            # Tools return ready-formatted templates. If one did, prefer it over
            # the model's paraphrase - the model tends to re-wrap it in markdown
            # that WhatsApp renders as literal asterisks.
            for tool_result in self.last_tool_results:
                tool_output = tool_result.get("result")
                if not tool_output:
                    continue
                if _is_template_response(str(tool_output)):
                    logger.debug(
                        "[Template] Using pre-formatted output from %s",
                        tool_result.get("tool_name"),
                    )
                    return str(tool_output)

            if "Agent stopped due to iteration limit or time limit" in str(result):
                metrics.incr("agent.iteration_limit")
                logger.warning("[Agent] Hit max iterations for message: %s", message[:100])
                return _ITERATION_LIMIT_REPLIES.get(intent, _ITERATION_LIMIT_REPLIES["default"])

            return response

        except Exception as agent_error:
            error_str = str(agent_error)
            metrics.incr("agent.invoke.error")
            logger.error("[Agent] Execution error: %s", error_str)

            if "validation error for PlaceOrderInput" in error_str and "items" in error_str:
                logger.warning("[Agent] PlaceOrderInput validation error - items field missing")
                if self._language_of(message) == "english":
                    return (
                        "To place your order, please specify the exact products and "
                        "quantities you want. For example: 'Product ID 1, quantity 2'."
                    )
                return (
                    "ඔබගේ ඇණවුම සදහා කරුණාකර නිශ්චිත නිෂ්පාදන සහ ප්‍රමාණ සඳහන් කරන්න. "
                    "උදා: 'නිෂ්පාදන අංක 1, ප්‍රමාණ 2'."
                )

            return self._fallback_message(message, technical=False)

        finally:
            # Record usage even on failure: a turn that errored after three tool
            # round trips still cost money, and hiding that would make the cost
            # reports understate exactly the turns worth investigating.
            self.last_usage = usage_callback.usage

    def _language_of(self, message: str) -> str:
        """Best-effort language guess for picking fallback copy."""
        try:
            from agent.language_agent import get_language_agent

            detected = get_language_agent().detect_language_simple(message)
            return "english" if detected not in ("sinhala", "singlish") else detected
        except Exception:
            # Sinhala has its own Unicode block - a direct check beats failing.
            return "sinhala" if re.search(r"[඀-෿]", message or "") else "english"

    def _fallback_message(self, message: str, technical: bool) -> str:
        language = self._language_of(message)
        if language in ("sinhala", "singlish"):
            return (
                "මට තාක්ෂණික ගැටලුවක් ඇත. කරුණාකර නැවත උත්සාහ කරන්න."
                if technical
                else "මට ඔබේ ඉල්ලීම සම්පූර්ණ කිරීමට අපහසුයි. කරුණාකර සරල වචන වලින් නැවත උත්සාහ කරන්න."
            )
        return (
            "I'm experiencing technical difficulties. Please try again."
            if technical
            else "I'm having trouble completing your request. Please try rephrasing it in simpler terms."
        )


_ITERATION_LIMIT_REPLIES = {
    "place_order": (
        "I'm working on processing your order. Let me help you step by step. "
        "Could you please confirm what items you'd like to order?"
    ),
    "order_tracking": "I can help you track your order. Please provide your order ID.",
    "product_info": (
        "I can help you with product information. What product would you like to know about?"
    ),
    "default": (
        "I'm here to help! Could you please rephrase your request or be more specific "
        "about what you need?"
    ),
}


def create_optimized_chatbot(seller_id: str, user_id: str) -> OptimizedChatbot:
    """Factory function to create optimized chatbot.

    Building one is not cheap (20 tools + a bound LLM + an executor). Prefer
    services.session_store.session_store.get_or_create, which reuses instances.
    """
    return OptimizedChatbot(seller_id, user_id)


def create_multi_agent_system(seller_id: str, user_id: str):
    """Optimized replacement for the original multi-agent system"""
    chatbot = create_optimized_chatbot(seller_id, user_id)

    def process_input(input_data, external_chat_history: Optional[list] = None):
        message = input_data.get("input", "")
        return chatbot.process_message(message, external_chat_history)

    return {"executor": process_input, "chat_history": chatbot.chat_history}
