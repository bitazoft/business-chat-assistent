
from langchain_openai import ChatOpenAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from db.database import SessionLocal
from models.schemas import Product, Order, ChatLog, OrderItem
from vector_store.vector_store import vector_store
import os
import numpy as np
import requests
from datetime import datetime
from typing import List

# Load environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

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
            product = db.query(Product).filter(Product.id == int(item["product_id"]), Product.seller_id == int(seller_id)).first()
            if not product or product.stock < item["quantity"]:
                db.rollback()
                return f"Product {item['product_id']} unavailable or insufficient stock"
            total_amount += product.price * item["quantity"]
            order_item = OrderItem(order_id=order.id, product_id=product.id, price=product.price, quantity=item["quantity"])
            db.add(order_item)
            product.stock -= item["quantity"]
        order.total_amount = total_amount
        db.commit()
        return f"Order placed successfully. Order ID: {order.id}"
    except Exception as e:
        db.rollback()
        return f"Error placing order: {str(e)}"
    finally:
        db.close()

def log_query(query: str, intent: str, entities: str, response: str, seller_id: str) -> None:
    db = SessionLocal()
    try:
        chat_log = ChatLog(
            user_query=query,
            intent=intent,
            entities=entities,
            response=response,
            seller_id=int(seller_id)
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

tools = [
    Tool(name="get_product_info", func=get_product_info, description="Get product details by name"),
    Tool(name="track_order", func=track_order, description="Track order by order ID"),
    Tool(name="place_order", func=place_order, description="Place an order for multiple products by seller ID, user ID, and items (list of {product_id, quantity})"),
    Tool(name="log_query", func=log_query, description="Log user queries with intent and entities"),
    Tool(name="query_context", func=query_context, description="Search product/FAQ documents")
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a business assistant. Use tools to fetch product info, track orders, or place orders. Log all queries. For general questions, use RAG context if relevant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)