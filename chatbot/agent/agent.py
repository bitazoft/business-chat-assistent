from langchain_openai import ChatOpenAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from db.database import SessionLocal
import pandas as pd
from utils.logger import get_logger,GlobalLogger
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import asyncio
import time
from functools import lru_cache
import threading
from typing import Union, List, Dict, Optional

# Get logger for this module
logger = get_logger(__name__)

# Import existing tools
from repositories.tools import (
    get_product_info,
    track_order,
    place_order,
    log_query,
    get_user_info,
    update_user_info,
    check_user_exists,
    save_user,
    create_tmp_user_id,
    get_all_products,
    add_item_to_order,
    remove_item_from_order,
    update_item_quantity_in_order,
    replace_order_items,
    get_all_orders_for_customer,
    get_pending_orders,
    get_order_details,
    check_product_stock,
    upload_payment_proof_and_update_order
)
from vector_store.vector_store import fast_vector_store as vector_store
from agent.customer_service_rag import customer_service_rag
from agent.language_agent import get_language_agent, detect_language_detailed,detect_language
import os
import re

# Load environment variables
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE", "https://api.deepseek.com/v1")

if(os.getenv("CHAT_MODEL")== "DEEPSEEK"):
    # Configure LLM with optimized settings for speed
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=API_KEY,
        base_url=API_BASE,
        temperature=0.1,  # Lower temperature for faster, more deterministic responses
        max_tokens=512,   # Limit response length for speed
        timeout=300,     
        max_retries=3     # Reduce retries for faster failure handling
    )
    
elif(os.getenv("CHAT_MODEL")=="GPT"):
    llm = ChatOpenAI(
        model_name=os.getenv("CHAT_MODEL_NAME","gpt-3.5-turbo"),
        temperature=0.1,  # Lower temperature for faster, more deterministic responses
        max_tokens=512,   # Limit response length for speed
        timeout=300,     
        max_retries=3     # Reduce retries for faster failure handling
    )


# Caching for frequently accessed data
@lru_cache(maxsize=100)
def get_cached_rag_examples(user_input: str, seller_id: str, k: int = 2,threshold: float = 3):
    """Cached RAG examples with reduced k for speed"""
    try:
        # Simplified RAG - only get essential examples
        results = vector_store.similarity_search(user_input, k=k, threshold=threshold)
        examples = []
        
        for result in results:
            if hasattr(result, 'metadata'):
                result_seller = result.metadata.get('seller_id') or result.metadata.get('category')
                result_intent = result.metadata.get('intent')  # Extract intent from metadata
                examples.append(f"{result.page_content[:200]}... (Intent: {result_intent})")  # Include intent in the examples
                    
        return "\n".join(examples[:k]) if examples else ""
    except Exception as e:
        logger.warning(f"[RAG Cache] Error: {str(e)}")
        return ""

@lru_cache(maxsize=50)
def get_cached_intents(seller_id: str):
    """Cached intent retrieval"""
    try:
        # Use hardcoded intents for speed instead of RAG lookup
        return "product_info, order_tracking, place_order, user_management, general_inquiry"
    except Exception as e:
        logger.warning(f"[Intent Cache] Error: {str(e)}")
        return "product_info, order_tracking, place_order, user_management, general_inquiry"

def get_unified_system_prompt(seller_id: str) -> str:
    """Get unified system prompt that handles all languages"""
    return f"""You are a business assistant for seller {seller_id}. You can communicate in English, Sinhala, and Singlish (mixed Sinhala-English).

            LANGUAGE ADAPTATION RULES:
            - Detect the user's language from their message
            - Respond in the SAME language style the user is using
            - If user writes in English: Respond in English
            - If user writes in Sinhala (සිංහල): Respond in Sinhala
            - If user writes in Singlish (mixed): Respond in Sinhala

            Available tools: {', '.join(['get_product_info', 'track_order', 'place_order', 'get_user_info', 'save_user', 'check_user_exists', 'update_user_info', 'add_item_to_order', 'remove_item_from_order', 'update_item_quantity_in_order', 'replace_order_items', 'get_all_orders_for_customer', 'get_pending_orders'])}

            CORE INSTRUCTIONS (Be direct and efficient):
            1. Product information: use get_product_info without image urls
            2. Order tracking: use track_order immediately with order ID
            3. Place orders: ALWAYS check_user_exists first, then proceed
            4. User management: use appropriate user tools
            5. Order management: use granular order editing tools
            6. Be helpful and match the user's communication style
            7. Execute tools directly - don't ask for confirmation unless user data is missing

            CRITICAL ORDER WORKFLOW (Must follow exactly):
            When user wants to place an order:
            1. FIRST: Execute check_user_exists (no confirmation needed)
            2. IF user does NOT exist:
            - Ask for details in user's language:
                * English: "To place your order, I need your details. Please provide your full name, email address, physical address, and phone number."
                * Sinhala: "ඔබගේ ඇණවුම සිදු කිරීම සදහා ඔබගේ විස්තර අවශ්‍යයි. කරුණාකර ඔබගේ සම්පූර්ණ නම, ඊමේල් ලිපිනය, ගෘහ ලිපිනය සහ දුරකථන අංකය ලබා දෙන්න."
                * Singlish: "ඔබගේ ඇණවුම සිදු කිරීම සදහා ඔබගේ විස්තර අවශ්‍යයි. කරුණාකර ඔබගේ සම්පූර්ණ නම, ඊමේල් ලිපිනය, ගෘහ ලිපිනය සහ දුරකථන අංකය ලබා දෙන්න."
            3. IF user exists: Use get_user_info and proceed with place_order
            4. Execute place_order immediately after user confirmation and ask payment method COD or Bank Transfer
            5. If he said bank transfer check whether the image received of the bank transfer confirmation receipt and confirm that is the bank transfer receipt and save that receipt image by running upload_payment_proof_and_update_order method with given file path

            ORDER EDITING (Execute immediately when requested):
            - add_item_to_order: Add new item to existing pending order
            - remove_item_from_order: Remove specific item from pending order  
            - update_item_quantity_in_order: Change quantity of existing item
            - replace_order_items: Replace all items (like old edit_order)
            - get_pending_orders: View user's pending orders
            - get_all_orders_for_customer: View all user's orders
            - Always request order ID if not provided, then execute immediately

            EFFICIENCY RULES:
            - Execute tools immediately when you have required parameters
            - Don't ask for confirmation unless absolutely necessary
            - NEVER generate fake user information
            - Extract parameters accurately from user input
            - If a tool fails, provide a clear explanation and alternative
            - Keep responses concise and actionable"""

# Fast intent detection using keywords instead of LLM
def fast_intent_detection(user_input: str) -> str:
    """Fast rule-based intent detection for multiple languages"""
    user_input_lower = user_input.lower()
    
    # Order tracking keywords (English, Sinhala, Singlish)
    order_tracking_keywords = [
        # English
        'track', 'order', 'status', 'delivery', 'shipped', 'tracking',
        
    ]
    
    # Place order keywords
    place_order_keywords = [
        # English
        'buy', 'purchase', 'order', 'cart', 'checkout', 'want to buy',
        
    ]
    
    # Product info keywords
    product_info_keywords = [
        # English
        'product', 'price', 'cost', 'available', 'stock', 'details', 'info',
       
    ]
    
    # User management keywords
    user_management_keywords = [
        # English
        'profile', 'account', 'update', 'change', 'personal', 'details',
       
    ]
    
    # Check keywords
    if any(keyword in user_input_lower for keyword in order_tracking_keywords):
        return "order_tracking"
    
    if any(keyword in user_input_lower for keyword in place_order_keywords):
        return "place_order"
    
    if any(keyword in user_input_lower for keyword in product_info_keywords):
        return "product_info"
    
    if any(keyword in user_input_lower for keyword in user_management_keywords):
        return "user_management"
    
    # Default to general inquiry
    return "general_inquiry"

class OptimizedChatbot:
    """Optimized single-agent chatbot for faster responses"""
    
    def __init__(self, seller_id: str, user_id: str):
        self.seller_id = seller_id
        self.user_id = user_id
        self.chat_history = []
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        
        # Track tool results for analytics (without changing existing logs)
        self.last_tool_results = []  # List of {"tool_name": str, "result": str}
        
    def _create_tools(self):
        """Create optimized tools with embedded context"""
        
        # Define input schemas
        class GetProductInfoInput(BaseModel):
            product_name: str = Field(description="Name of the product")

        class TrackOrderInput(BaseModel):
            order_id: str = Field(description="Order ID to track")

        class PlaceOrderInput(BaseModel):
            items: List[dict] = Field(description="List of items with product_id and quantity")

        class SaveUserInput(BaseModel):
            name: str = Field(description="User's full name")
            email: str = Field(description="User's email address")
            address: str = Field(description="User's address")
            number: str = Field(description="User's phone number")

        class UpdateUserInfoInput(BaseModel):
            name: str = Field(description="User's name", default="")
            email: str = Field(description="User's email", default="")
            address: str = Field(description="User's address", default="")
            number: str = Field(description="User's phone", default="")
            
        class OrderItemInput(BaseModel):
            product_id: Union[int, str] = Field(..., description="Product ID or Name")
            quantity: int = Field(..., gt=0, description="Quantity of the product")

        class EditOrderInput(BaseModel):
            customer_id: str = Field(..., description="User ID who placed the order")
            order_id: Union[str, int] = Field(..., description="Order ID to be edited")
            new_items: List[OrderItemInput] = Field(..., description="Updated list of order items")
        
        class AddItemToOrderInput(BaseModel):
            order_id: str = Field(..., description="Order ID to add item to")
            product_identifier: str = Field(..., description="Product ID (numeric) or product name (string)")
            quantity: int = Field(..., gt=0, description="Quantity to add")

        class RemoveItemFromOrderInput(BaseModel):
            order_id: str = Field(..., description="Order ID to remove item from")
            product_identifier: str = Field(..., description="Product ID (numeric) or product name (string)")

        class UpdateItemQuantityInput(BaseModel):
            order_id: str = Field(..., description="Order ID to update item in")
            product_identifier: str = Field(..., description="Product ID (numeric) or product name (string)")
            new_quantity: int = Field(..., gt=0, description="New quantity for the item")

        class ReplaceOrderItemsInput(BaseModel):
            order_id: str = Field(..., description="Order ID to replace items in")
            new_items: List[OrderItemInput] = Field(..., description="New list of order items")
        
        class GetOrdersInput(BaseModel):
            customer_id: str = Field(..., description="User ID to retrieve orders for")
            
        class GetOrderDetailsInput(BaseModel):
            order_id: Union[int, str] = Field(..., description="Order ID to retrieve")

        class CheckStockInput(BaseModel):
            product_id: Union[int, str] = Field(..., description="Product ID to check")
            quantity: int = Field(..., gt=0, description="Quantity to verify against stock")
        class UploadPaymentProofInput(BaseModel):
            order_id: str = Field(..., description="Order ID to upload payment proof for")
            payment_proof_file: str = Field(..., description="Payment proof file location")

        class EmptyInput(BaseModel):
            pass

        # Wrapper functions with context and result tracking
        def get_product_info_wrapper(product_name: str) -> dict:
            result = get_product_info(seller_id=self.seller_id, product_name=product_name)
            self.last_tool_results.append({"tool_name": "get_product_info", "result": str(result)})
            return result

        def track_order_wrapper(order_id: str) -> dict:
            result = track_order(order_id=order_id)
            self.last_tool_results.append({"tool_name": "track_order", "result": str(order_id)})
            return result

        def place_order_wrapper(items: List[dict]) -> dict:
            result = place_order(seller_id=self.seller_id, user_id=self.user_id, items=items)
            # self.last_tool_results.append({"tool_name": "place_order", "result": str(result)})
            return result

        def save_user_wrapper(name: str, email: str, address: str, number: str) -> dict:
            result = save_user(user_id=self.user_id, name=name, email=email, address=address, number=number)
            # self.last_tool_results.append({"tool_name": "save_user", "result": str(result)})
            return result

        def get_user_info_wrapper() -> dict:
            result = get_user_info(user_id=self.user_id)
            # self.last_tool_results.append({"tool_name": "get_user_info", "result": str(result)})
            return result

        def check_user_exists_wrapper() -> bool:
            result = check_user_exists(user_id=self.user_id)
            # self.last_tool_results.append({"tool_name": "check_user_exists", "result": str(result)})
            return result

        def get_all_products_wrapper() -> List[str]:
            result = get_all_products(seller_id=self.seller_id)
            # self.last_tool_results.append({"tool_name": "get_all_products", "result": str(result)})
            return result

        def update_user_info_wrapper(name: str = "", email: str = "", address: str = "", number: str = "") -> dict:
            name = None if not name else name
            email = None if not email else email
            address = None if not address else address
            number = None if not number else number
            result = update_user_info(user_id=self.user_id, name=name, email=email, address=address, number=number)
            # self.last_tool_results.append({"tool_name": "update_user_info", "result": str(result)})
            return result
        
        def add_item_to_order_wrapper(order_id: str, product_identifier: str, quantity: int) -> str:
            result = add_item_to_order(
                customer_id=self.user_id,
                order_id=order_id,
                product_identifier=product_identifier,
                quantity=quantity
            )
            # self.last_tool_results.append({"tool_name": "add_item_to_order", "result": str(result)})
            return result

        def remove_item_from_order_wrapper(order_id: str, product_identifier: str) -> str:
            result = remove_item_from_order(
                customer_id=self.user_id,
                order_id=order_id,
                product_identifier=product_identifier
            )
            # self.last_tool_results.append({"tool_name": "remove_item_from_order", "result": str(result)})
            return result

        def update_item_quantity_in_order_wrapper(order_id: str, product_identifier: str, new_quantity: int) -> str:
            result = update_item_quantity_in_order(
                customer_id=self.user_id,
                order_id=order_id,
                product_identifier=product_identifier,
                new_quantity=new_quantity
            )
            # self.last_tool_results.append({"tool_name": "update_item_quantity_in_order", "result": str(result)})
            return result

        def replace_order_items_wrapper(order_id: str, new_items: List[dict]) -> str:
            result = replace_order_items(
                customer_id=self.user_id,
                order_id=order_id,
                new_items=new_items
            )
            # self.last_tool_results.append({"tool_name": "replace_order_items", "result": str(result)})
            return result
        
        def get_all_orders_for_customer_wrapper() -> list:
            result = get_all_orders_for_customer(customer_id=self.user_id)
            # self.last_tool_results.append({"tool_name": "get_all_orders_for_customer", "result": str(result)})
            return result

        def get_pending_orders_wrapper() -> list:
            result = get_pending_orders(customer_id=self.user_id)
            # self.last_tool_results.append({"tool_name": "get_pending_orders", "result": str(result)})
            return result
        
        def get_order_details_wrapper(order_id: int) -> dict:
            result = get_order_details(order_id=order_id)
            self.last_tool_results.append({"tool_name": "get_order_details", "result": str(order_id)})
            return result

        def check_product_stock_wrapper(product_id: int, quantity: int) -> dict:
            result = check_product_stock(product_id=product_id, quantity=quantity)
            self.last_tool_results.append({"tool_name": "check_product_stock", "result": str(product_id)})
            return result

        def upload_payment_proof_and_update_order_wrapper(order_id: str, payment_proof_file: str) -> str:
            result = upload_payment_proof_and_update_order(
                order_id=order_id,
                file_path=payment_proof_file
            )
            # self.last_tool_results.append({"tool_name": "upload_payment_proof_and_update_order", "result": str(result)})
            return result

        # Create tools
        return [
            StructuredTool(
                name="get_product_info",
                func=get_product_info_wrapper,
                description="Get product details by name",
                args_schema=GetProductInfoInput
            ),
            StructuredTool(
                name="track_order",
                func=track_order_wrapper,
                description="Track order status by order ID",
                args_schema=TrackOrderInput
            ),
            StructuredTool(
                name="place_order",
                func=place_order_wrapper,
                description="Place an order with list of items",
                args_schema=PlaceOrderInput
            ),
            StructuredTool(
                name="save_user",
                func=save_user_wrapper,
                description="Create new user with details",
                args_schema=SaveUserInput
            ),
            StructuredTool(
                name="get_user_info",
                func=get_user_info_wrapper,
                description="Get current user information",
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="check_user_exists",
                func=check_user_exists_wrapper,
                description="Check if user exists",
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="update_user_info",
                func=update_user_info_wrapper,
                description="Update user information",
                args_schema=UpdateUserInfoInput
            ),
            StructuredTool(
                name="get_all_products",
                func=get_all_products_wrapper,
                description="Get all products for seller",
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="add_item_to_order",
                func=add_item_to_order_wrapper,
                description="Add an item to an existing pending order or update quantity if item already exists",
                args_schema=AddItemToOrderInput
            ),
            StructuredTool(
                name="remove_item_from_order",
                func=remove_item_from_order_wrapper,
                description="Remove an item completely from an existing pending order",
                args_schema=RemoveItemFromOrderInput
            ),
            StructuredTool(
                name="update_item_quantity_in_order",
                func=update_item_quantity_in_order_wrapper,
                description="Update the quantity of a specific item in an existing pending order",
                args_schema=UpdateItemQuantityInput
            ),
            StructuredTool(
                name="replace_order_items",
                func=replace_order_items_wrapper,
                description="Replace all items in an existing pending order with new items (like original edit_order)",
                args_schema=ReplaceOrderItemsInput
            ),
            StructuredTool(
                name="get_all_orders_for_customer",
                description="Get all orders and their items for the current customer",
                func=get_all_orders_for_customer_wrapper,
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="get_pending_orders",
                description="Retrieve all pending orders for the current customer",
                func=get_pending_orders_wrapper,
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="get_order_details",
                description="Get detailed information about a specific order.",
                func=get_order_details_wrapper,
                args_schema=GetOrderDetailsInput
            ),
            StructuredTool(
                name="check_product_stock",
                description="Check if a product has enough stock before editing.",
                func=check_product_stock_wrapper,
                args_schema=CheckStockInput
            ),
            StructuredTool(
                name="upload_payment_proof_and_update_order",
                func=upload_payment_proof_and_update_order_wrapper,
                description="Upload payment proof for an order and update its status.",
                args_schema=UploadPaymentProofInput
            )
        ]
    
    def _create_agent(self):
        """Create optimized single agent with unified language support"""
        llm_with_tools = llm.bind_tools(self.tools)
        
        # Unified prompt that handles all languages with dynamic examples
        prompt = ChatPromptTemplate.from_messages([

            ("system", get_unified_system_prompt(self.seller_id) + "\n\nContext Examples: {examples}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        agent = create_openai_tools_agent(llm_with_tools, self.tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=False,
            max_iterations=10,  # Increased from 3 to 10 to handle complex workflows
            early_stopping_method="generate",  # Allow agent to stop early if it has an answer
            handle_parsing_errors=True,
            return_intermediate_steps=False  # Don't return intermediate steps for cleaner output
        )
    
    def get_tool_results(self) -> List[Dict[str, str]]:
        """Get the tool results from the last conversation turn"""
        return self.last_tool_results.copy()
    
    def clear_tool_results(self):
        """Clear tool results"""
        self.last_tool_results = []

    def extract_id_from_str(self, result_data, id_type) -> str:
        """Extract ID from result data based on type"""
        result_str = str(result_data)
        
        # Check if result is a dictionary with product_id key
        if isinstance(result_data, dict) and id_type in result_data:
            return str(result_data[id_type])
        
        # Extract from formatted string "Product ID: 123, Product: ..."
        match = re.search(rf'{id_type}: (\d+)', result_str)
        if match:
            return match.group(1)
        
        # If no match found, return empty string
        return ""

    def get_img_urls(self) -> List[str]:
        """Get image URLs from the last tool results"""
        result = self.get_tool_results()
        img_urls = []
        for item in result:
           tool_name = item.get("tool_name", "")
           if tool_name == "get_product_info":
               # Split by comma first, then clean up each URL
               urls_text = item['result']
               if isinstance(urls_text, str):
                img_urls.extend(re.findall(r'https?://[^\s,]+', str(urls_text)))                   
        return img_urls

    def extract_entities(self) -> Dict[str, Any]:
        """Extract entities from the last tool results"""
        result = self.get_tool_results()
        entities = {}
        for item in result:
            tool_name = item.get("tool_name", "")
            if tool_name == "get_product_info":
                # Use the dedicated method to extract product ID
                product_id = self.extract_id_from_str(item['result'], id_type='Product ID')
                if product_id:
                    entities["product_id"] = product_id
            elif tool_name == "track_order":
                entities["order_id"] = item['result']
            elif tool_name == "get_order_details":
                entities["order_id"] = item['result']
            elif tool_name == "check_product_stock":
                entities["product_id"] = item['result']
        return entities

    def log_query(self, query: str, intent: str, response: str, entities: Union[str, Dict[str, Any], List] = "", response_time: int = 0):
        """Log query asynchronously to avoid blocking"""

        try:
            log_query(
                query=query,
                intent=intent,
                entities=entities,
                response=response,
                seller_id=self.seller_id,
                user_id=self.user_id,
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"[Optimized] logging error: {str(e)}")
    
    def process_message(self, message: str, external_chat_history: List[Dict] = None) -> str:
        """Process message with language detection and optimizations"""
        start_time = time.time()
        
        # Clear previous tool results
        self.clear_tool_results()
        
        try:
            # Use external chat history if provided
            if external_chat_history:
                self.chat_history = external_chat_history
            
            # Add user message
            self.chat_history.append({"role": "user", "content": message})
            
            # Detect language from user message using language agent
            # language_agent = get_language_agent()
            # language_result = detect_language_detailed(message)
            detected_language = "unknown"
            if(os.getenv("LANGUAGE_DETECTION_ENABLED","false").lower()=="true"):
                detected_language = detect_language(message)
                logger.info(f"[Language] Detected language: {detected_language}")

            # Fast intent detection (skip LLM call)
            intent = fast_intent_detection(message)
            logger.info(f"[Optimized] Detected intent: {intent} for message: '{message}' in {time.time() - start_time:.2f}s")
            
            # # Get minimal RAG examples
            examples = None
            if(os.getenv("RAG_ENABLED","false").lower()=="true"):
                examples = get_cached_rag_examples(message, self.seller_id, k=3)
                logger.info(f"[Optimized] Retrieved RAG examples: {examples}... for intent: {intent}")

            # Format chat history for agent
            formatted_history = []
            for msg in self.chat_history[-20:]:  # Only last 3 exchanges for speed
                if msg["role"] == "user":
                    formatted_history.append(("human", msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(("assistant", msg["content"]))
            
            # Execute agent with unified prompt that includes examples
            # The prompt will automatically detect language and respond appropriately
            try:
                result = self.agent.invoke({
                    "input": message,
                    "examples": examples if examples is not None else "",
                    "intent": intent,
                    "chat_history": formatted_history
                })
                
                response = result.get("output", "I couldn't process your request.")
                
                # Check if the agent stopped due to max iterations
                if "Agent stopped due to iteration limit or time limit" in str(result):
                    logger.warning(f"[Agent] Hit max iterations for message: {message}")
                    # Try to provide a helpful response based on intent
                    if intent == "place_order":
                        response = "I'm working on processing your order. Let me help you step by step. Could you please confirm what items you'd like to order?"
                    elif intent == "order_tracking":
                        response = "I can help you track your order. Please provide your order ID."
                    elif intent == "product_info":
                        response = "I can help you with product information. What product would you like to know about?"
                    else:
                        response = "I'm here to help! Could you please rephrase your request or be more specific about what you need?"
                        
            except Exception as agent_error:
                logger.error(f"[Agent] Execution error: {str(agent_error)}")
                # Fallback response based on detected language
                language_agent = get_language_agent()
                error_language = language_agent.detect_language_simple(message)
                if error_language == 'sinhala':
                    response = "මට ඔබේ ඉල්ලීම සම්පූර්ණ කිරීමට අපහසුයි. කරුණාකර සරල වචන වලින් නැවත උත්සාහ කරන්න."
                elif error_language == 'singlish':
                    response = "මට ඔබේ ඉල්ලීම සම්පූර්ණ කිරීමට අපහසුයි. කරුණාකර සරල වචන වලින් නැවත උත්සාහ කරන්න."
                else:
                    response = "I'm having trouble completing your request. Please try rephrasing it in simpler terms."
            
            # Add assistant response
            self.chat_history.append({"role": "assistant", "content": response})
            
            # Limit chat history
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            
            total_time = time.time() - start_time
            logger.info(f"[Optimized] Total processing time: {total_time:.2f}s, Language: {detected_language}")

            self.log_query(message, intent, response, self.extract_entities(), total_time*1000)

            return response
            
        except Exception as e:
            logger.error(f"[Optimized] Error: {str(e)}")
            # Return error message in appropriate language
            language_agent = get_language_agent()
            error_language = language_agent.detect_language_simple(message)
            if error_language == 'sinhala':
                return "මට තාක්ෂණික ගැටලුවක් ඇත. කරුණාකර නැවත උත්සාහ කරන්න."
            elif error_language == 'singlish':
                return "මට තාක්ෂණික ගැටලුවක් ඇත. කරුණාකර නැවත උත්සාහ කරන්න."
            else:
                return "I'm experiencing technical difficulties. Please try again."

def create_optimized_chatbot(seller_id: str, user_id: str) -> OptimizedChatbot:
    """Factory function to create optimized chatbot"""
    return OptimizedChatbot(seller_id, user_id)

# Backward compatibility wrapper
def create_multi_agent_system(seller_id: str, user_id: str):
    """Optimized replacement for the original multi-agent system"""
    chatbot = create_optimized_chatbot(seller_id, user_id)
    
    def process_input(input_data, external_chat_history: list = None):
        message = input_data.get("input", "")
        return chatbot.process_message(message, external_chat_history)
    
    return {"executor": process_input, "chat_history": chatbot.chat_history}
