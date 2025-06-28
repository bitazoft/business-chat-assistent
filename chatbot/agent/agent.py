from langchain_openai import ChatOpenAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from db.database import SessionLocal
from models.schemas import Product, Order, ChatLog, OrderItem, User
from vector_store.vector_store import vector_store
import os
import numpy as np
import requests
from datetime import datetime
from typing import List
from functools import partial

# Load environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# Make functions available for import
__all__ = ['create_agent_executor', 'log_query', 'get_product_info', 'track_order', 'place_order', 'get_user_info', 'update_user_info', 'check_user_exists', 'save_user', 'create_tmp_user_id']

# LangChain Tools
def get_product_info(product_name: str, seller_id: str) -> str:
    db = SessionLocal()
    try:
        product = db.query(Product).filter(Product.name.ilike(f"%{product_name}%"), Product.seller_id == int(seller_id)).first()
        if product:
            return f"Product: {product.name}, Description: {product.description}, Price: ${product.price}, Stock: {product.stock}"
        return "Product not found"
    finally:
        db.close()

def track_order(order_id: str) -> str:
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.id == int(order_id)).first()
        if order:
            return f"Order ID: {order.id}, Status: {order.status}, Created: {order.created_at}"
        return "Order not found"
    finally:
        db.close()

def place_order(seller_id: str, user_id: str, items: List[dict]) -> str:
    db = SessionLocal()
    try:
        total_amount = 0
        order = Order(seller_id=int(seller_id), user_id=user_id, status="pending", total_amount=0)
        db.add(order)
        db.flush()  # Get order.id before committing
        for item in items:
            product = None
            
            # Check if product_id is numeric (ID) or string (name)
            product_identifier = item["product_id"]
            
            if str(product_identifier).isdigit():
                # Look up by product ID
                product = db.query(Product).filter(Product.id == int(product_identifier), Product.seller_id == int(seller_id)).first()
            else:
                # Look up by product name
                product = db.query(Product).filter(Product.name.ilike(f"%{product_identifier}%"), Product.seller_id == int(seller_id)).first()
            
            if not product:
                db.rollback()
                return f"Product '{product_identifier}' not found"
            
            if product.stock < item["quantity"]:
                db.rollback()
                return f"Product '{product.name}' has insufficient stock. Available: {product.stock}, Requested: {item['quantity']}"
            
            total_amount += product.price * item["quantity"]
            order_item = OrderItem(order_id=order.id, product_id=product.id, price=product.price, quantity=item["quantity"])
            db.add(order_item)
            product.stock -= item["quantity"]
        order.total_amount = total_amount
        db.commit()
        return f"Order placed successfully. Order ID: {order.id}, Total Amount: ${total_amount:.2f}"
    except Exception as e:
        db.rollback()
        return f"Error placing order: {str(e)}"
    finally:
        db.close()

def check_user_exists(user_id: str) -> bool:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        return user is not None
    finally:
        db.close()       

def get_user_info(user_id: str) -> str:
    """Get user information from database"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            return f"User ID: {user.id}, Name: {user.name}, Email: {user.email}, Address: {user.address}, Phone: {user.number}"
        return "User not found"
    finally:
        db.close()

def update_user_info(user_id: str, name: str = None, email: str = None, address: str = None, number: str = None) -> str:
    """Update user information in database"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return "User not found"
        
        # Update only provided fields
        if name is not None:
            user.name = name
        if email is not None:
            user.email = email
        if address is not None:
            user.address = address
        if number is not None:
            user.number = number
            
        db.commit()
        return f"User information updated successfully. Updated details: Name: {user.name}, Email: {user.email}, Address: {user.address}, Phone: {user.number}"
    except Exception as e:
        db.rollback()
        return f"Error updating user information: {str(e)}"
    finally:
        db.close()

def create_tmp_user_id() -> str:
    """Create a temporary user ID based on current timestamp and random number"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_number = np.random.randint(1000, 9999)
    return f"user_{timestamp}_{random_number}"         

def save_user(user_id: str, name: str, email: str, address: str, number: str) -> None:
    db = SessionLocal()
    try:
        user = User(id=user_id, name=name, email=email, address=address, number=number)
        db.add(user)
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def log_query(query: str, intent: str, entities: str, response: str, seller_id: str, user_id: str) -> None:
    db = SessionLocal()
    try:
        chat_log = ChatLog(
            user_query=query,
            intent=intent,
            entities=entities,
            response=response,
            seller_id=int(seller_id),
            user_id=user_id
        )
        db.add(chat_log)
        db.commit()
    finally:
        db.close()

def query_context(query: str, seller_id: str) -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": query, "model": "deepseek-embedding"}  # Replace with DeepSeek's embedding model
    response = requests.post(f"{DEEPSEEK_API_BASE}/embeddings", json=payload, headers=headers)
    response.raise_for_status()
    query_embedding = np.array(response.json()["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)
    results = vector_store.search(query_embedding, seller_id)
    return "\n".join(results)

# LangChain Agent Setup
llm = ChatDeepSeek(
    model="deepseek-chat",  # Replace with actual DeepSeek model name
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE
)

def create_agent_executor(seller_id: str, user_id: str):
    """Create an agent executor with seller_id and user_id bound to tools"""
    
    # Create tools with bound seller_id and user_id
    tools = [
        Tool(
            name="get_product_info", 
            func=partial(get_product_info, seller_id=seller_id), 
            description="Get product details by name. Required parameters: product_name (string)"
        ),
        Tool(
            name="track_order", 
            func=track_order, 
            description="Track order status by order ID. Required parameters: order_id (string)"
        ),
        Tool(
            name="place_order", 
            func=partial(place_order, seller_id=seller_id, user_id=user_id), 
            description="Place an order for multiple products. Required parameters: items (list of dictionaries with product_id and quantity keys). product_id can be either numeric ID or product name string."
        ),
        Tool(
            name="log_query", 
            func=partial(log_query, seller_id=seller_id, user_id=user_id), 
            description="Log user queries with intent, entities, and response. Required parameters: query (string), intent (string), entities (string), response (string)"
        ),
        Tool(
            name="save_user",
            func=partial(save_user, user_id=user_id),
            description="Create a new user with provided details. Required parameters: name (string), email (string), address (string), number (string)"
        ),
        Tool(
            name="get_user_info",
            func=partial(get_user_info, user_id=user_id),
            description="Retrieve user information from database. No additional parameters required - user_id is automatically provided."
        ),
        Tool(
            name="check_user_exists",
            func=partial(check_user_exists, user_id=user_id),
            description="Check if user exists in database. No additional parameters required - user_id is automatically provided."
        ),
        Tool(
            name="update_user_info",
            func=partial(update_user_info, user_id=user_id),
            description="Update user information in database. Parameters: name (string, optional), email (string, optional), address (string, optional), number (string, optional). Only provided parameters will be updated."
        )
        # Tool(name="query_context", func=partial(query_context, seller_id=seller_id), description="Search product/FAQ documents using RAG. Required parameters: query (string)")
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a business assistant for seller ID: {seller_id}. Use tools to fetch product info, track orders, or place orders. The seller_id and user_id are automatically provided to tools that need them. 

        USER MANAGEMENT WORKFLOW:
        - For general inquiries (product info, order tracking, questions): No need to collect user details
        - ONLY when user wants to PLACE AN ORDER:
          1. Check if user exists using check_user_exists
          2. If user doesn't exist, ask for their details (name, email, address, phone number) and create them using save_user
          3. If user exists, get their info using get_user_info and show it to confirm details are correct
          4. If user wants to update any information, use update_user_info with only the fields they want to change
          5. Only proceed with placing order after confirming user details

        GENERAL INSTRUCTIONS:
        - Log all queries using log_query
        - For product questions, use get_product_info (no user details needed)
        - For order tracking, use track_order (no user details needed)
        - For placing orders, use place_order (user details required first)
        - Always be helpful and only ask for information when necessary"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Default agent executor (will be replaced by create_agent_executor)
# tools = [
#     Tool(name="get_product_info", func=get_product_info, description="Get product details by name and seller ID. Required parameters: product_name (string), seller_id (string)"),
#     Tool(name="track_order", func=track_order, description="Track order status by order ID. Required parameters: order_id (string)"),
#     Tool(name="place_order", func=place_order, description="Place an order for multiple products. Required parameters: seller_id (string), user_id (string), items (list of dictionaries with product_id and quantity keys)"),
#     Tool(name="log_query", func=log_query, description="Log user queries with intent, entities, response, seller ID and user ID. Required parameters: query (string), intent (string), entities (string), response (string), seller_id (string), user_id (string)"),
#     # Tool(name="query_context", func=query_context, description="Search product/FAQ documents using RAG. Required parameters: query (string), seller_id (string)")
# ]

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a business assistant for seller ID: {seller_id}. Use tools to fetch product info, track orders, or place orders. Always include the seller_id parameter when calling tools. Log all queries. For general questions, use RAG context if relevant."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])

# agent = create_openai_tools_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)