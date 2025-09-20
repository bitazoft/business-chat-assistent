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
    upload_payment_proof_and_update_order,
    cancel_order
)
from vector_store.vector_store import fast_vector_store as vector_store
from agent.customer_service_rag import customer_service_rag
from agent.language_agent import get_language_agent, detect_language_detailed,detect_language
import os
import re

# Import templates
from templates.message_templates import MessageTemplates

# Template integration helper functions
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

# Load environment variables
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE", "https://api.deepseek.com/v1")

if(os.getenv("AI_PROVIDER")== "DEEPSEEK"):
    # Configure LLM with optimized settings for speed
    llm = ChatDeepSeek(
        model=os.getenv("CHAT_MODEL","deepseek-chat"),
        api_key=API_KEY,
        base_url=API_BASE,
        temperature=0.1,  # Lower temperature for faster, more deterministic responses
        max_tokens=512,   # Limit response length for speed
        timeout=300,     
        max_retries=3     # Reduce retries for faster failure handling
    )
    
elif(os.getenv("AI_PROVIDER")=="GPT"):
    llm = ChatOpenAI(
        model_name=os.getenv("CHAT_MODEL","gpt-3.5-turbo"),
        openai_api_key=API_KEY,  # Explicitly set API key
        temperature=0.1,  # Lower temperature for faster, more deterministic responses
        max_tokens=256,   # Limit response length for speed
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
    tools_list = ', '.join(['get_product_info', 'track_order', 'place_order', 'get_user_info', 'save_user', 'check_user_exists', 'update_user_info', 'add_item_to_order', 'remove_item_from_order', 'update_item_quantity_in_order', 'replace_order_items', 'get_all_orders_for_customer', 'get_pending_orders', 'cancel_order'])
    return f"""You are a business assistant for seller {seller_id}.

            LANGUAGE ADAPTATION RULES:
            - Detect the user's language from their message
            - Respond in the SAME language style the user is using
            - If user writes in English: Respond in English
            - If user writes in Sinhala (සිංහල): Respond in Sinhala
            - If user writes in Singlish (mixed): Respond in Sinhala
            - But if he ask order details or products show the retrieved data in English(as it is in the database)

            Available tools: {tools_list}

            CORE INSTRUCTIONS (Be direct and efficient):
            1. Product information: use get_product_info without image urls
            2. Order tracking: use track_order immediately with order ID
            3. Place orders: ALWAYS check_user_exists first, then proceed
            4. User management: use appropriate user tools
            5. Order management: use granular order editing tools
            6. Be helpful and match the user's communication style
            7. Execute tools directly - don't ask for confirmation unless user data is missing

            CRITICAL RESPONSE FORMATTING:
            - When tools return beautifully formatted responses with emojis and templates, return ONLY the tool output - NO ADDITIONAL FORMATTING
            - NEVER add asterisks (**) or markdown formatting to tool responses that are already formatted
            - DO NOT reformat, summarize, rephrase, or change the tool output in any way
            - If a tool returns a structured template with emojis (🛍️, 📦, 🚚, etc.), return it exactly as-is
            - DO NOT create your own version of the formatted response - just return the tool output directly
            - Only add brief commentary if the tool output needs translation or context, but keep the original formatting intact

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
            - Ask for confirmation for putting , cancelling orders
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
        
        # Define input schemas with comprehensive examples
        class GetProductInfoInput(BaseModel):
            product_name: str = Field(
                description="Name or identifier of the product to get information about",
                examples=["laptop", "iPhone 15", "gaming chair", "wireless headphones", "1"]
            )

        class TrackOrderInput(BaseModel):
            order_id: str = Field(
                description="Order ID to track - can be numeric string or alphanumeric",
                examples=["12345", "ORD001", "67890", "ORDER_ABC123"]
            )

        class PlaceOrderInput(BaseModel):
            items: List[dict] = Field(
                description="List of items to order. Each item should be a dictionary with 'product_id' (int) and 'quantity' (int) keys. Example: [{'product_id': 1, 'quantity': 2}, {'product_id': 3, 'quantity': 1}] use get productInfo method to get the product ids",
                examples=[[{"product_id": 1, "quantity": 2}], [{"product_id": 3, "quantity": 1}, {"product_id": 2, "quantity": 3}]]
            )

        class SaveUserInput(BaseModel):
            name: str = Field(
                description="User's full name (first and last name)",
                examples=["John Smith", "Nimal Perera", "Sarah Johnson"]
            )
            email: str = Field(
                description="User's email address in valid email format", 
                examples=["john.smith@gmail.com", "nimal@yahoo.com", "sarah.j@outlook.com"]
            )
            address: str = Field(
                description="User's complete physical address including city/area",
                examples=["123 Main St, Colombo 07", "45/2 Galle Road, Mount Lavinia", "78 Kandy Road, Peradeniya"]
            )
            number: str = Field(
                description="User's phone number (mobile or landline)",
                examples=["+94771234567", "0771234567", "011-2345678", "+94112345678"]
            )

        class UpdateUserInfoInput(BaseModel):
            name: str = Field(
                description="User's name to update (optional, leave empty if not changing)", 
                default="",
                examples=["John Smith", "පීතර සිල්වා", ""]
            )
            email: str = Field(
                description="User's email to update (optional, leave empty if not changing)", 
                default="",
                examples=["john.new@gmail.com", "updated@email.com", ""]
            )
            address: str = Field(
                description="User's address to update (optional, leave empty if not changing)", 
                default="",
                examples=["New Address, Colombo 05", "456 Updated St, Kandy", ""]
            )
            number: str = Field(
                description="User's phone to update (optional, leave empty if not changing)", 
                default="",
                examples=["+94779876543", "0119876543", ""]
            )
            
        class OrderItemInput(BaseModel):
            product_id: Union[int, str] = Field(
                ..., 
                description="Product ID (numeric) or product name (string)",
                examples=[1, "laptop", 25, "iPhone 15", "gaming_mouse"]
            )
            quantity: int = Field(
                ..., 
                gt=0, 
                description="Quantity of the product (must be positive integer)",
                examples=[1, 2, 5, 10, 3]
            )

        class EditOrderInput(BaseModel):
            customer_id: str = Field(
                ..., 
                description="User ID who placed the order",
                examples=["user123", "customer_456", "USR789"]
            )
            order_id: Union[str, int] = Field(
                ..., 
                description="Order ID to be edited (numeric or string)",
                examples=[12345, "ORD001", 67890, "ORDER_ABC"]
            )
            new_items: List[OrderItemInput] = Field(
                ..., 
                description="Updated complete list of order items (replaces existing items)"
            )
        
        class AddItemToOrderInput(BaseModel):
            order_id: str = Field(
                ..., 
                description="Order ID to add item to",
                examples=["12345", "ORD001", "67890"]
            )
            product_identifier: str = Field(
                ..., 
                description="Product ID (numeric) or product name (string)",
                examples=["1", "laptop", "25", "iPhone 15", "gaming_mouse"]
            )
            quantity: int = Field(
                ..., 
                gt=0, 
                description="Quantity to add (positive integer)",
                examples=[1, 2, 3, 5, 10]
            )

        class RemoveItemFromOrderInput(BaseModel):
            order_id: str = Field(
                ..., 
                description="Order ID to remove item from",
                examples=["12345", "ORD001", "67890"]
            )
            product_identifier: str = Field(
                ..., 
                description="Product ID (numeric) or product name (string) to remove",
                examples=["1", "laptop", "25", "iPhone 15"]
            )

        class UpdateItemQuantityInput(BaseModel):
            order_id: str = Field(
                ..., 
                description="Order ID to update item quantity in",
                examples=["12345", "ORD001", "67890"]
            )
            product_identifier: str = Field(
                ..., 
                description="Product ID (numeric) or product name (string) to update",
                examples=["1", "laptop", "25", "iPhone 15"]
            )
            new_quantity: int = Field(
                ..., 
                gt=0, 
                description="New quantity for the item (positive integer)",
                examples=[1, 2, 5, 10, 15]
            )

        class ReplaceOrderItemsInput(BaseModel):
            order_id: str = Field(
                ..., 
                description="Order ID to completely replace items in",
                examples=["12345", "ORD001", "67890"]
            )
            new_items: List[OrderItemInput] = Field(
                ..., 
                description="Complete new list of order items (replaces all existing items)"
            )
        
        class GetOrdersInput(BaseModel):
            customer_id: str = Field(
                ..., 
                description="User ID to retrieve all orders for",
                examples=["user123", "customer_456", "USR789"]
            )
            
        class GetOrderDetailsInput(BaseModel):
            order_id: Union[int, str] = Field(
                ..., 
                description="Specific Order ID to retrieve detailed information for",
                examples=[12345, "ORD001", 67890, "ORDER_ABC123"]
            )

        class CheckStockInput(BaseModel):
            product_id: Union[int, str] = Field(
                ..., 
                description="Product ID (numeric) or name (string) to check stock for",
                examples=[1, "laptop", 25, "iPhone 15", "gaming_mouse"]
            )
            quantity: int = Field(
                ..., 
                gt=0, 
                description="Quantity to verify if available in stock",
                examples=[1, 2, 5, 10, 20]
            )
        
        class UploadPaymentProofInput(BaseModel):
            order_id: str = Field(
                ..., 
                description="Order ID to upload payment proof for",
                examples=["12345", "ORD001", "67890"]
            )
            payment_proof_file: str = Field(
                ..., 
                description="File path to payment proof image (bank transfer receipt, etc.)",
                examples=["/uploads/payment_123.jpg", "payment_receipt.png", "/tmp/bank_transfer.pdf"]
            )
        
        class CancelOrderInput(BaseModel):
            order_id: str = Field(
                ..., 
                description="Order ID to cancel (only pending orders can be cancelled)",
                examples=["12345", "ORD001", "67890"]
            )
            reason: str = Field(
                default="", 
                description="Optional reason for cancellation",
                examples=["Changed mind", "Found better price", "No longer needed", "Ordered by mistake", ""]
            )

        class EmptyInput(BaseModel):
            pass

        # Wrapper functions with context and result tracking
        def get_product_info_wrapper(product_name: str) -> str:
            result = get_product_info(seller_id=self.seller_id, product_name=product_name)
            self.last_tool_results.append({"tool_name": "get_product_info", "result": result.get("images", [])})
            
            # Format result using beautiful template
            if "not found" in str(result).lower():
                return format_error_response("not_found", f"Product '{product_name}' not found in our catalog")
            else:
                return format_product_info_response(result)

        def track_order_wrapper(order_id: str) -> str:
            result = track_order(order_id=order_id)
            self.last_tool_results.append({"tool_name": "track_order", "result": str(result)})
            
            # Format result using beautiful template
            if "not found" in str(result).lower() or "invalid" in str(result).lower():
                return format_error_response("not_found", f"Order #{order_id} not found")
            else:
                # Parse the result and format it
                tracking_data = {"order_id": order_id, "status": "pending", "details": result}
                return format_tracking_response(tracking_data)

        def place_order_wrapper(items: List[dict]) -> str:
            # Validate items parameter
            if not items or not isinstance(items, list):
                logger.error(f"[place_order_wrapper] Invalid items parameter: {items}")
                return format_error_response("invalid_input", "Invalid items parameter. Expected list of dictionaries with product_id and quantity.")
            
            # Validate each item in the list
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    logger.error(f"[place_order_wrapper] Item {i} is not a dictionary: {item}")
                    return format_error_response("invalid_input", f"Item {i} must be a dictionary with product_id and quantity keys.")
                
                if 'product_id' not in item or 'quantity' not in item:
                    logger.error(f"[place_order_wrapper] Item {i} missing required keys: {item}")
                    return format_error_response("invalid_input", f"Item {i} must have both 'product_id' and 'quantity' keys.")
            
            result = place_order(seller_id=self.seller_id, user_id=self.user_id, items=items)
            # self.last_tool_results.append({"tool_name": "place_order", "result": str(result)})
            
            # Format result using beautiful template
            if "error" in str(result).lower() or "not found" in str(result).lower():
                return format_error_response("system_error", str(result))
            elif "insufficient stock" in str(result).lower():
                return format_error_response("out_of_stock", str(result))
            else:
                # Format as order confirmation
                order_data = {"items": items, "status": "pending", "result": result}
                return format_order_details_response(order_data)

        def save_user_wrapper(name: str, email: str, address: str, number: str) -> dict:
            result = save_user(user_id=self.user_id, name=name, email=email, address=address, number=number)
            # self.last_tool_results.append({"tool_name": "save_user", "result": str(result)})
            return result

        def get_user_info_wrapper() -> str:
            result = get_user_info(user_id=self.user_id)
            # self.last_tool_results.append({"tool_name": "get_user_info", "result": str(result)})
            
            # Format result using beautiful template
            if "not found" in str(result).lower() or "does not exist" in str(result).lower():
                return format_error_response("not_found", "User information not found")
            else:
                return format_customer_info_response(result)

        def check_user_exists_wrapper() -> bool:
            result = check_user_exists(user_id=self.user_id)
            # self.last_tool_results.append({"tool_name": "check_user_exists", "result": str(result)})
            return result

        def get_all_products_wrapper() -> str:
            result = get_all_products(seller_id=self.seller_id)
            # self.last_tool_results.append({"tool_name": "get_all_products", "result": str(result)})
            
            # Format result using beautiful template
            if not result or "no products" in str(result).lower():
                return format_error_response("not_found", "No products available at the moment")
            else:
                return format_product_list_response(result)

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
            if not result or "not found" in str(result).lower():
                return format_error_response("not_found", f"Order #{order_id} not found")
            else:
                return format_order_details_response(result)
            

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

        def cancel_order_wrapper(order_id: str, reason: str = "") -> str:
            result = cancel_order(
                customer_id=self.user_id,
                order_id=order_id,
                reason=reason
            )
            # self.last_tool_results.append({"tool_name": "cancel_order", "result": str(result)})
            return result

        # Create tools with comprehensive descriptions and examples
        return [
            StructuredTool(
                name="get_product_info",
                func=get_product_info_wrapper,
                description="Get detailed product information including price, description, and images by product name or ID. Example: get_product_info(product_name='laptop') or get_product_info(product_name='iPhone 15')",
                args_schema=GetProductInfoInput
            ),
            StructuredTool(
                name="track_order",
                func=track_order_wrapper,
                description="Track order status and delivery information by order ID. Returns current status, estimated delivery, tracking details. Example: track_order(order_id='12345') or track_order(order_id='ORD001')",
                args_schema=TrackOrderInput
            ),
            StructuredTool(
                name="place_order",
                func=place_order_wrapper,
                description="Place a new order with specified items and quantities. Each item must have 'product_id' and 'quantity'. Example: place_order(items=[{'product_id': 1, 'quantity': 2}, {'product_id': 'laptop', 'quantity': 1}]). NEVER use empty dict {}!",
                args_schema=PlaceOrderInput
            ),
            StructuredTool(
                name="save_user",
                func=save_user_wrapper,
                description="Create a new user account with personal details (name, email, address, phone). Required for first-time customers. Example: save_user(name='John Smith', email='john@email.com', address='123 Main St', number='+94771234567')",
                args_schema=SaveUserInput
            ),
            StructuredTool(
                name="get_user_info",
                func=get_user_info_wrapper,
                description="Retrieve current user's profile information (name, email, address, phone). No parameters needed - uses current user context. Example: get_user_info()",
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="check_user_exists",
                func=check_user_exists_wrapper,
                description="Check if current user exists in the system. Returns True/False. Essential before placing orders. No parameters needed. Example: check_user_exists()",
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="update_user_info",
                func=update_user_info_wrapper,
                description="Update specific user profile fields. Only provide fields to change, leave others empty. Example: update_user_info(name='New Name', email='') to only update name",
                args_schema=UpdateUserInfoInput
            ),
            StructuredTool(
                name="get_all_products",
                func=get_all_products_wrapper,
                description="Get complete list of all available products from the seller's catalog. No parameters needed. Example: get_all_products()",
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="add_item_to_order",
                func=add_item_to_order_wrapper,
                description="Add a new item to existing pending order or increase quantity if item exists. Example: add_item_to_order(order_id='12345', product_identifier='laptop', quantity=1)",
                args_schema=AddItemToOrderInput
            ),
            StructuredTool(
                name="remove_item_from_order",
                func=remove_item_from_order_wrapper,
                description="Completely remove a specific item from existing pending order. Example: remove_item_from_order(order_id='12345', product_identifier='laptop')",
                args_schema=RemoveItemFromOrderInput
            ),
            StructuredTool(
                name="update_item_quantity_in_order",
                func=update_item_quantity_in_order_wrapper,
                description="Change quantity of existing item in pending order. Example: update_item_quantity_in_order(order_id='12345', product_identifier='laptop', new_quantity=3)",
                args_schema=UpdateItemQuantityInput
            ),
            StructuredTool(
                name="replace_order_items",
                func=replace_order_items_wrapper,
                description="Replace ALL items in pending order with completely new item list. Example: replace_order_items(order_id='12345', new_items=[OrderItemInput(product_id=1, quantity=2)])",
                args_schema=ReplaceOrderItemsInput
            ),
            StructuredTool(
                name="get_all_orders_for_customer",
                description="Retrieve complete order history for current customer including all statuses (pending, confirmed, delivered). No parameters needed. Example: get_all_orders_for_customer()",
                func=get_all_orders_for_customer_wrapper,
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="get_pending_orders",
                description="Get only pending/unpaid orders for current customer. Useful for order editing. No parameters needed. Example: get_pending_orders()",
                func=get_pending_orders_wrapper,
                args_schema=EmptyInput
            ),
            StructuredTool(
                name="get_order_details",
                description="Get detailed information about specific order including items, status, total cost. Example: get_order_details(order_id='12345') or get_order_details(order_id=67890)",
                func=get_order_details_wrapper,
                args_schema=GetOrderDetailsInput
            ),
            StructuredTool(
                name="check_product_stock",
                description="Verify if sufficient stock available before placing/editing orders. Example: check_product_stock(product_id=1, quantity=5) or check_product_stock(product_id='laptop', quantity=2)",
                func=check_product_stock_wrapper,
                args_schema=CheckStockInput
            ),
            StructuredTool(
                name="upload_payment_proof_and_update_order",
                func=upload_payment_proof_and_update_order_wrapper,
                description="Upload bank transfer receipt image and update order payment status. Example: upload_payment_proof_and_update_order(order_id='12345', payment_proof_file='/uploads/receipt.jpg')",
                args_schema=UploadPaymentProofInput
            ),
            StructuredTool(
                name="cancel_order",
                func=cancel_order_wrapper,
                description="Cancel a pending order and restore product stock. Only pending orders can be cancelled. Example: cancel_order(order_id='12345', reason='Changed mind')",
                args_schema=CancelOrderInput
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
        img_urls = []
        for item in self.get_tool_results():
            if item.get("tool_name") == "get_product_info":
                img_urls.extend(item['result'])
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
        """Log query synchronously"""
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
                
                # Post-process response to ensure template formatting is preserved
                tool_results = self.get_tool_results()
                if tool_results:
                    # Look for template-formatted tool responses
                    for tool_result in tool_results:
                        tool_name = tool_result.get("tool_name", "")
                        tool_output = str(tool_result.get("result", ""))
                        
                        # Check if this is a formatted template response (starts with emojis)
                        template_indicators = ["🛍️", "🚚", "📋", "🛒", "👤", "💰", "🔍", "📦", "⚠️", "🔧", "💳"]
                        if any(tool_output.startswith(indicator) for indicator in template_indicators):
                            # This is a pre-formatted template - use it directly instead of agent response
                            response = tool_output
                            logger.info(f"[Template] Using pre-formatted template response for {tool_name}")
                            break
                        
                        # Also check if agent response contains duplicate formatting
                        if ("**" in response and "*" in tool_output and 
                            tool_name in ["get_product_info", "track_order", "place_order", "get_all_products", "get_user_info"]):
                            # Agent added extra formatting - use the clean tool output
                            response = tool_output
                            logger.info(f"[Template] Replaced agent's duplicated formatting with clean template for {tool_name}")
                            break
                
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
                error_str = str(agent_error)
                logger.error(f"[Agent] Execution error: {error_str}")
                
                # Handle specific validation errors
                if "validation error for PlaceOrderInput" in error_str and "items" in error_str:
                    logger.warning(f"[Agent] PlaceOrderInput validation error detected - items field missing")
                    language_agent = get_language_agent()
                    error_language = language_agent.detect_language_simple(message)
                    if error_language == 'sinhala':
                        response = "ඔබගේ ඇණවුම සදහා කරුණාකර නිශ්චිත නිෂ්පාදන සහ ප්‍රමාණ සඳහන් කරන්න. උදා: 'නිෂ්පාදන අංක 1, ප්‍රමාණ 2'."
                    else:
                        response = "To place your order, please specify the exact products and quantities you want. For example: 'Product ID 1, quantity 2'."
                else:
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

            # Schedule background logging using threading to avoid blocking response
            def background_log():
                try:
                    self.log_query(message, intent, response, self.extract_entities(), total_time*1000)
                except Exception as log_error:
                    logger.error(f"Background logging error: {str(log_error)}")
            
            # Submit to background thread (non-blocking)
            threading.Thread(target=background_log, daemon=True).start()

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
