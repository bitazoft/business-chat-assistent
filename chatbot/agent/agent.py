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
    # edit_order
    get_all_orders_for_customer,
    get_pending_orders,
    get_order_details,
    check_product_stock,
    edit_order_with_stock_update
)
from vector_store.vector_store import fast_vector_store as vector_store
from agent.customer_service_rag import customer_service_rag
import os

# Load environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# Configure LLM with optimized settings for speed
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
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

# Fast intent detection using keywords instead of LLM
def fast_intent_detection(user_input: str) -> str:
    """Fast rule-based intent detection"""
    user_input_lower = user_input.lower()
    
    # Order tracking keywords
    if any(keyword in user_input_lower for keyword in ['track', 'order', 'status', 'delivery', 'shipped']):
        return "order_tracking"
    
    # Place order keywords
    if any(keyword in user_input_lower for keyword in ['buy', 'purchase', 'order', 'cart', 'checkout']):
        return "place_order"
    
    # Product info keywords
    if any(keyword in user_input_lower for keyword in ['product', 'price', 'cost', 'available', 'stock', 'details']):
        return "product_info"
    
    # User management keywords
    if any(keyword in user_input_lower for keyword in ['profile', 'account', 'update', 'change', 'personal']):
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
        
        class GetOrdersInput(BaseModel):
            customer_id: str = Field(..., description="User ID to retrieve orders for")
            
        class GetOrderDetailsInput(BaseModel):
            order_id: Union[int, str] = Field(..., description="Order ID to retrieve")

        class CheckStockInput(BaseModel):
            product_id: Union[int, str] = Field(..., description="Product ID to check")
            quantity: int = Field(..., gt=0, description="Quantity to verify against stock")


        class EmptyInput(BaseModel):
            pass

        # Wrapper functions with context
        def get_product_info_wrapper(product_name: str) -> dict:
            return get_product_info(seller_id=self.seller_id, product_name=product_name)

        def track_order_wrapper(order_id: str) -> dict:
            return track_order(order_id=order_id)

        def place_order_wrapper(items: List[dict]) -> dict:
            return place_order(seller_id=self.seller_id, user_id=self.user_id, items=items)

        def save_user_wrapper(name: str, email: str, address: str, number: str) -> dict:
            return save_user(user_id=self.user_id, name=name, email=email, address=address, number=number)

        def get_user_info_wrapper() -> dict:
            return get_user_info(user_id=self.user_id)

        def check_user_exists_wrapper() -> bool:
            return check_user_exists(user_id=self.user_id)

        def get_all_products_wrapper() -> List[str]:
            return get_all_products(seller_id=self.seller_id)

        def update_user_info_wrapper(name: str = "", email: str = "", address: str = "", number: str = "") -> dict:
            name = None if not name else name
            email = None if not email else email
            address = None if not address else address
            number = None if not number else number
            return update_user_info(user_id=self.user_id, name=name, email=email, address=address, number=number)
        
        # def edit_order_wrapper(customer_id: str, order_id: str, new_items: List[dict]) -> str:
        #     return edit_order(
        #         customer_id=customer_id,
        #         order_id=order_id,
        #         new_items=new_items
        #     )
        
        def get_all_orders_for_customer_wrapper(customer_id: str) -> list:
            return get_all_orders_for_customer(customer_id=customer_id)

        def get_pending_orders_wrapper(customer_id: str) -> list:
            return get_pending_orders(customer_id=customer_id)
        
        def get_order_details_wrapper(order_id: int) -> dict:
            return get_order_details(order_id=order_id)

        def check_product_stock_wrapper(product_id: int, quantity: int) -> dict:
            return check_product_stock(product_id=product_id, quantity=quantity)

        def edit_order_with_stock_update_wrapper(order_id: int, customer_id: str, new_items: list[dict]) -> dict:
            return edit_order_with_stock_update(order_id=order_id, customer_id=customer_id, new_items=new_items)

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
            # StructuredTool(
            #     name="edit_order",
            #     func=edit_order_wrapper,
            #     description="Edit an existing pending order by specifying new items and their quantities",
            #     args_schema=EditOrderInput
            # ),
            StructuredTool(
                name="get_all_orders_for_customer",
                description="Get all orders and their items for a customer by ID.",
                func=get_all_orders_for_customer_wrapper,
                args_schema=GetOrdersInput
            ),
            StructuredTool(
                name="get_pending_orders",
                description="Retrieve all pending orders for a specific customer.",
                func=get_pending_orders_wrapper,
                args_schema=GetOrdersInput
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
                name="edit_order_with_stock_update",
                description="Edit a pending order and update stock in a single transaction.",
                func=edit_order_with_stock_update_wrapper,
                args_schema=EditOrderInput
            )
        ]
    
    def _create_agent(self):
        """Create optimized single agent"""
        llm_with_tools = llm.bind_tools(self.tools)
        
        # Simplified prompt for faster processing
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a fast business assistant for seller {self.seller_id}.

                Available tools: {[tool.name for tool in self.tools]}

                Instructions:
                1. For product info: use get_product_info
                2. For order tracking: use track_order  
                3. For placing orders: check_user_exists first, then place_order
                4. For user management: use appropriate user tools
                5. For editing orders: use edit_order and related tools (see below)
                6. Be concise and direct

                CRITICAL USER DATA WORKFLOW FOR ORDERS:
                When user wants to place an order, follow this EXACT sequence:
                1. FIRST: Always use check_user_exists to verify if user exists
                2. IF user does NOT exist:
                - NEVER create fake or assumed user data
                - NEVER use save_user without explicit user-provided information
                - Ask the user: "To place your order, I need your details. Please provide your full name, email address, physical address, and phone number."
                - WAIT for user response with their actual details
                - ONLY then use save_user with the information they provided
                3. IF user exists: Use get_user_info to show their details and confirm they're correct
                4. ONLY after user confirmation, proceed with place_order

                EDITING ORDER FLOW:
                Use this exact sequence when user wants to modify or change an existing order:
                1. ALWAYS confirm or extract the order ID from user
                2. If user does not remember the order ID, use get_pending_orders to show editable orders for the customer
                3. If order ID is found, use get_order_details to display order items
                4. Ask user: "Please provide the updated items (product name or ID and quantity) you'd like to add or change in your order."
                5. Once updated item info is received:
                - Internally call check_product_stock for each item to verify stock is available
                - If all stock is sufficient, call edit_order to update the order and reduce product stock
                - If any item is out of stock, notify the user and cancel the edit attempt

                IMPORTANT RULES:
                - Only allow editing if order status is 'pending' (use get_pending_orders to verify)
                - NEVER assume missing order ID or product details
                - NEVER edit an order without both valid ID and item list from user
                - NEVER proceed if stock is not available — validate first
                - NEVER generate, assume, or create fake user information
                - NEVER use save_user without explicit user input containing name, email, address, phone
                - For product info and order tracking: No user details needed
                - For placing or editing orders: User details and/or order info are MANDATORY
                - Extract parameters accurately from user input and chat history
                - Execute tools directly; do not describe actions without executing

                Context: {{examples}}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(llm_with_tools, self.tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,  # Disable verbose for speed
            max_iterations=3,  # Limit iterations
            handle_parsing_errors=True
        )
    
    def process_message(self, message: str, external_chat_history: List[Dict] = None) -> str:
        """Process message with optimizations"""
        start_time = time.time()
        
        try:
            # Use external chat history if provided
            if external_chat_history:
                self.chat_history = external_chat_history
            
            # Add user message
            self.chat_history.append({"role": "user", "content": message})
            
            # Fast intent detection (skip LLM call)
            intent = fast_intent_detection(message)
            logger.info(f"[Optimized] Detected intent: {intent} for message: '{message}' in {time.time() - start_time:.2f}s")
            
            # Get minimal RAG examples
            examples = get_cached_rag_examples(message, self.seller_id, k=1)
            logger.info(f"[Optimized] Retrieved RAG examples: {examples}... for intent: {intent}")
            # Format chat history for agent
            formatted_history = []
            for msg in self.chat_history[-6:]:  # Only last 3 exchanges for speed
                if msg["role"] == "user":
                    formatted_history.append(("human", msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(("assistant", msg["content"]))
            
            # Execute agent (single LLM call)
            result = self.agent.invoke({
                "input": message,
                "examples": examples,
                "intent": intent,
                "chat_history": formatted_history
            })
            
            response = result.get("output", "I couldn't process your request.")
            
            # Add assistant response
            self.chat_history.append({"role": "assistant", "content": response})
            
            # Limit chat history
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            # Log query asynchronously (don't wait)
            threading.Thread(target=self._log_query_async, args=(message, intent, response)).start()
            
            total_time = time.time() - start_time
            logger.info(f"[Optimized] Total processing time: {total_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"[Optimized] Error: {str(e)}")
            return "I'm experiencing technical difficulties. Please try again."
    
    def _log_query_async(self, query: str, intent: str, response: str):
        """Log query asynchronously to avoid blocking"""
        try:
            log_query(
                query=query,
                intent=intent,
                entities="fast_detected",
                response=response,
                seller_id=self.seller_id,
                user_id=self.user_id
            )
        except Exception as e:
            logger.error(f"[Optimized] Async logging error: {str(e)}")

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
