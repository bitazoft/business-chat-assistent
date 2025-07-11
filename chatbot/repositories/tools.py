from db.database import SessionLocal
from models.schemas import Product, Order, ChatLog, OrderItem, User
from vector_store.vector_store import vector_store
import os
import numpy as np
import requests
from datetime import datetime
from typing import List

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

def get_all_products(seller_id: str) -> List[str]:
    db = SessionLocal()
    try:
        products = db.query(Product).filter(Product.seller_id == int(seller_id)).all()
        if products:
            return [f"Product: {p.name}, Price: ${p.price}, Stock: {p.stock}" for p in products]
        return ["No products found for this seller"]
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

def save_user(user_id: str, name: str, email: str, address: str, number: str) -> str:
    db = SessionLocal()
    try:
        user = User(id=user_id, name=name, email=email, address=address, number=number)
        db.add(user)
        db.commit()
        return f"User successfully created: {name} ({email}). Account ID: {user_id}"
    except Exception as e:
        db.rollback()
        return f"Error creating user: {str(e)}"
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
