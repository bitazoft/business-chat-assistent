from db.database import SessionLocal
from models.schemas import Product, Order, ChatLog, OrderItem, Customers
from vector_store.vector_store import vector_store
import os
import numpy as np
import requests
from datetime import datetime
from typing import List
from sqlalchemy import text
from collections import defaultdict

# LangChain Tools
def get_product_info(product_name: str, seller_id: str) -> str:
    db = SessionLocal()
    try:
        product = db.query(Product).filter(Product.name.ilike(f"%{product_name}%"), Product.sellerId == int(seller_id)).first()
        if product:
            return f"Product: {product.name}, Description: {product.description}, Price: ${product.price}, Stock: {product.stock}"
        return "Product not found"
    finally:
        db.close()

def get_all_products(seller_id: str) -> List[str]:
    db = SessionLocal()
    try:
        products = db.query(Product).filter(Product.sellerId == int(seller_id)).all()
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
        order = Order(sellerId=int(seller_id), customerId=user_id, status="pending", total_amount=0)
        db.add(order)
        db.flush()  # Get order.id before committing
        for item in items:
            product = None
            
            # Check if product_id is numeric (ID) or string (name)
            product_identifier = item["product_id"]
            
            if str(product_identifier).isdigit():
                # Look up by product ID
                product = db.query(Product).filter(Product.id == int(product_identifier), Product.sellerId == int(seller_id)).first()
            else:
                # Look up by product name
                product = db.query(Product).filter(Product.name.ilike(f"%{product_identifier}%"), Product.sellerId == int(seller_id)).first()
            
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
        user = db.query(Customers).filter(Customers.id == user_id).first()
        return user is not None
    finally:
        db.close()       

def get_user_info(user_id: str) -> str:
    """Get user information from database"""
    db = SessionLocal()
    try:
        customer = db.query(Customers).filter(Customers.id == user_id).first()
        if customer:
            return f"User ID: {customer.id}, Name: {customer.name}, Email: {customer.email}, Address: {customer.address}, Phone: {customer.number1}"
        return "User not found"
    finally:
        db.close()

def update_user_info(user_id: str, name: str = None, email: str = None, address: str = None, number: str = None) -> str:
    """Update user information in database"""
    db = SessionLocal()
    try:
        customer = db.query(Customers).filter(Customers.id == user_id).first()
        if not customer:
            return "User not found"
        
        # Update only provided fields
        if name is not None:
            customer.name = name
        if email is not None:
            customer.email = email
        if address is not None:
            customer.address = address
        if number is not None:
            customer.number1 = number
            
        db.commit()
        return f"User information updated successfully. Updated details: Name: {customer.name}, Email: {customer.email}, Address: {customer.address}, Phone: {customer.number1}"
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
        customer = Customers(id=user_id, name=name, email=email, address=address, number1=number)
        db.add(customer)
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
            sellerId=int(seller_id),
            customerId=user_id
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

# def edit_order(customer_id: str, order_id: str, new_items: List[dict]) -> str:
#     """
#     Edit an existing pending order by user.
#     Each item in new_items must have:
#     - 'product_id' or 'name'
#     - 'quantity'
#     """
#     db = SessionLocal()
#     try:
#         # customer_id = user_id
#         order = db.query(Order).filter(Order.id == int(order_id)).first()
#         if not order:
#             return "Order not found"
#         if order.status != "pending":
#             return f"Order cannot be edited. Current status: {order.status}"

        
#         print("going to restock")
#         # Fetch existing order items and restock
#         existing_items = db.query(OrderItem).filter(OrderItem.order_id == order.id).all()
#         for item in existing_items:
#             product = db.query(Product).filter(Product.id == item.product_id).first()
#             if product:
#                 product.stock += item.quantity  # Restock
#             db.delete(item)  # Remove old items
#         print("Item restocked")

#         db.flush()

#         # Now add new items as per the updated list
#         total_amount = 0
#         for item in new_items:
#             product = None
#             identifier = item["product_id"]
            
#             if str(identifier).isdigit():
#                 product = db.query(Product).filter(Product.id == int(identifier), Product.seller_id == order.seller_id).first()
#             else:
#                 product = db.query(Product).filter(Product.name.ilike(f"%{identifier}%"), Product.seller_id == order.seller_id).first()
            
#             if not product:
#                 db.rollback()
#                 return f"Product '{identifier}' not found"
#             if product.stock < item["quantity"]:
#                 db.rollback()
#                 return f"Insufficient stock for product '{product.name}'. Available: {product.stock}, Requested: {item['quantity']}"

#             total_amount += product.price * item["quantity"]
#             order_item = OrderItem(order_id=order.id, product_id=product.id, price=product.price, quantity=item["quantity"])
#             db.add(order_item)
#             product.stock -= item["quantity"]

#         # Update the total and commit
#         order.total_amount = total_amount
#         db.commit()

#         return f"Order {order.id} successfully updated. New total: ${total_amount:.2f}"

#     except Exception as e:
#         db.rollback()
#         return f"Error editing order: {str(e)}"
#     finally:
#         db.close()

def get_all_orders_for_customer(customer_id: str) -> list:
    """Get all orders and items for a specific customer using raw SQL"""
    db = SessionLocal()
    try:
        # Raw SQL query
        sql = text("""
            SELECT 
                o.id AS order_id,
                o.status,
                o.total_amount,
                o.created_at,
                oi.quantity,
                oi.price,
                p.name AS product_name
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE o."customerId" = :customer_id
            ORDER BY o.created_at DESC
        """)

        rows = db.execute(sql, {"customer_id": customer_id}).mappings().fetchall()

        # Group items by order
        order_map = defaultdict(lambda: {
            "order_id": None,
            "status": None,
            "total_amount": None,
            "created_at": None,
            "items": []
        })

        for row in rows:
            order_id = row["order_id"]
            order_data = order_map[order_id]

            # Fill order meta info only once
            if order_data["order_id"] is None:
                order_data.update({
                    "order_id": row["order_id"],
                    "status": row["status"],
                    "total_amount": row["total_amount"],
                    "created_at": str(row["created_at"]),
                })

            # Append item info
            order_data["items"].append({
                "product": row["product_name"],
                "quantity": row["quantity"],
                "price": row["price"]
            })

        if order_map:
            return list(order_map.values())
        return [{"message": "No orders found for this customer"}]

    finally:
        db.close()

    
def get_pending_orders(customer_id: str) -> list:
    """Get all pending orders and items for a customer"""
    db = SessionLocal()
    try:
        sql = text("""
            SELECT 
                o.id AS order_id,
                o.status,
                o.total_amount,
                o.created_at,
                oi.quantity,
                oi.price,
                p.name AS product_name
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE o."customerId" = :customer_id
              AND o.status = 'pending'
            ORDER BY o.created_at DESC
        """)

        rows = db.execute(sql, {"customer_id": customer_id}).mappings().fetchall()

        order_map = defaultdict(lambda: {
            "order_id": None,
            "status": None,
            "total_amount": None,
            "created_at": None,
            "items": []
        })

        for row in rows:
            order_id = row["order_id"]
            order = order_map[order_id]

            if order["order_id"] is None:
                order.update({
                    "order_id": row["order_id"],
                    "status": row["status"],
                    "total_amount": row["total_amount"],
                    "created_at": str(row["created_at"]),
                })

            order["items"].append({
                "product": row["product_name"],
                "quantity": row["quantity"],
                "price": row["price"]
            })

        if order_map:
            return list(order_map.values())
        return [{"message": "No pending orders found"}]

    finally:
        db.close()

def get_order_details(order_id: int) -> dict:
    """Get detailed info for a specific order using raw SQL"""
    db = SessionLocal()
    try:
        sql = text("""
            SELECT 
                o.id AS order_id,
                o."customerId",
                o.status,
                o.total_amount,
                o.created_at,
                oi.quantity,
                oi.price,
                p.name AS product_name
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE o.id = :order_id
        """)

        rows = db.execute(sql, {"order_id": order_id}).mappings().fetchall()

        if not rows:
            return {"error": "Order not found"}

        # Use first row for order-level info
        order_info = rows[0]
        order_data = {
            "order_id": order_info["order_id"],
            "customer_id": order_info["customerId"],
            "status": order_info["status"],
            "total_amount": order_info["total_amount"],
            "created_at": str(order_info["created_at"]),
            "items": []
        }

        # Collect items
        for row in rows:
            order_data["items"].append({
                "product": row["product_name"],
                "quantity": row["quantity"],
                "price": row["price"]
            })

        return order_data

    finally:
        db.close()

def check_product_stock(product_id: int, quantity: int) -> dict:
    """Check if a product has enough stock"""
    db = SessionLocal()
    try:
        product = db.query(Products).filter(Products.id == product_id).first()
        if not product:
            return {"available": False, "stock": 0, "error": "Product not found"}
        return {
            "available": product.stock >= quantity,
            "stock": product.stock,
            "product": product.name
        }
    finally:
        db.close()

def edit_order_with_stock_update(order_id: int, customer_id: str, new_items: list[dict]) -> dict:
    """Edit a pending order and update product stock in a single transaction"""
    db = SessionLocal()
    try:
        order = db.query(Orders).filter(Orders.id == order_id).first()

        if not order:
            return {"success": False, "error": "Order not found"}
        if order.customerId != customer_id:
            return {"success": False, "error": "Order does not belong to this customer"}
        if order.status != "pending":
            return {"success": False, "error": "Only pending orders can be edited"}

        # Validate stock for all items first
        for item in new_items:
            product = db.query(Products).filter(Products.id == item["product_id"]).first()
            if not product:
                return {"success": False, "error": f"Product {item['product_id']} not found"}
            if product.stock < item["quantity"]:
                return {
                    "success": False,
                    "error": f"Insufficient stock for product {product.name}"
                }

        # Delete old order items
        db.query(OrderItems).filter(OrderItems.order_id == order_id).delete()

        total = 0
        for item in new_items:
            product = db.query(Products).filter(Products.id == item["product_id"]).first()

            # Create new order item
            order_item = OrderItems(
                order_id=order_id,
                product_id=item["product_id"],
                quantity=item["quantity"],
                price=product.price
            )
            db.add(order_item)

            # Adjust stock
            product.stock -= item["quantity"]

            total += product.price * item["quantity"]

        # Update order total
        order.total_amount = total

        db.commit()
        return {"success": True, "updated_order_id": order_id}
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    finally:
        db.close()
