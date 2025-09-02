from db.database import SessionLocal
from models.schemas import Product, Order, ChatLog, OrderItem, Customers, SellerProfile, ProductImage
from vector_store.vector_store import vector_store
import os
import numpy as np
import requests
import json
from datetime import datetime
from typing import List, Union, Dict, Any
from sqlalchemy import text
from collections import defaultdict
import asyncio

# Import S3 service for payment proof handling
try:
    from services.s3_service import upload_payment_proof, generate_presigned_url
except ImportError:
    # Fallback if boto3 is not installed
    async def upload_payment_proof(*args, **kwargs):
        return {"error": "S3 service not available - boto3 not installed"}
    
    async def generate_presigned_url(*args, **kwargs):
        return {"error": "S3 service not available - boto3 not installed"}

# LangChain Tools
def get_product_info(product_name: str, seller_id: str) -> str:
    db = SessionLocal()
    try:
        product = db.query(Product).filter(Product.name.ilike(f"%{product_name}%"), Product.seller_id == int(seller_id)).first()
        if product:
            imgs = get_product_images(product.id)
            return f"Product ID: {product.id}, Product: {product.name}, Description: {product.description}, Price: ${product.price}, Stock: {product.stock}, Images: {', '.join(imgs)}"
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
        order = Order(seller_id=int(seller_id), customer_id=user_id, status="pending", total_amount=0)
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

def log_query(query: str, intent: str, entities: Union[str, Dict[str, Any], List], response: str, seller_id: str, user_id: str, response_time : int) -> None:
    db = SessionLocal()
    try:
        # Convert entities to proper JSON format
        if isinstance(entities, str):
            # Try to parse as JSON first
            try:
                entities_json = json.loads(entities)
            except json.JSONDecodeError:
                # If parsing fails, treat as plain text and wrap in object
                entities_json = {"raw": entities}
        elif isinstance(entities, (dict, list)):
            entities_json = entities
        else:
            entities_json = {"raw": str(entities)}
            
        chat_log = ChatLog(
            user_query=query,
            intent=intent,
            entities=entities_json,
            response=response,
            seller_id=int(seller_id),
            customer_id=user_id,
            response_time_ms=response_time
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

def add_item_to_order(customer_id: str, order_id: str, product_identifier: str, quantity: int) -> str:
    """
    Add an item to an existing pending order or update quantity if item already exists.
    product_identifier can be product ID (numeric) or product name (string)
    """
    db = SessionLocal()
    try:
        # Validate order exists and belongs to customer
        order = db.query(Order).filter(Order.id == int(order_id), Order.customer_id == customer_id).first()
        if not order:
            return "Order not found or doesn't belong to this customer"
        if order.status != "pending":
            return f"Order cannot be edited. Current status: {order.status}"

        # Find the product
        product = None
        if str(product_identifier).isdigit():
            product = db.query(Product).filter(Product.id == int(product_identifier), Product.seller_id == order.seller_id).first()
        else:
            product = db.query(Product).filter(Product.name.ilike(f"%{product_identifier}%"), Product.seller_id == order.seller_id).first()
        
        if not product:
            return f"Product '{product_identifier}' not found for this seller"

        # Check if item already exists in order
        existing_item = db.query(OrderItem).filter(OrderItem.order_id == order.id, OrderItem.product_id == product.id).first()
        
        if existing_item:
            # Update existing item quantity
            old_quantity = existing_item.quantity
            new_quantity = quantity
            quantity_diff = new_quantity - old_quantity
            
            if product.stock < quantity_diff:
                return f"Insufficient stock for product '{product.name}'. Available: {product.stock}, Additional needed: {quantity_diff}"
            
            existing_item.quantity = new_quantity
            product.stock -= quantity_diff
            order.total_amount += product.price * quantity_diff
            
            action = f"Updated quantity for '{product.name}' from {old_quantity} to {new_quantity}"
        else:
            # Add new item to order
            if product.stock < quantity:
                return f"Insufficient stock for product '{product.name}'. Available: {product.stock}, Requested: {quantity}"
            
            order_item = OrderItem(order_id=order.id, product_id=product.id, price=product.price, quantity=quantity)
            db.add(order_item)
            product.stock -= quantity
            order.total_amount += product.price * quantity
            
            action = f"Added '{product.name}' (quantity: {quantity}) to order"

        db.commit()
        return f"{action}. Order total: ${order.total_amount:.2f}"

    except Exception as e:
        db.rollback()
        return f"Error adding item to order: {str(e)}"
    finally:
        db.close()

def remove_item_from_order(customer_id: str, order_id: str, product_identifier: str) -> str:
    """
    Remove an item completely from an existing pending order.
    product_identifier can be product ID (numeric) or product name (string)
    """
    db = SessionLocal()
    try:
        # Validate order exists and belongs to customer
        order = db.query(Order).filter(Order.id == int(order_id), Order.customer_id == customer_id).first()
        if not order:
            return "Order not found or doesn't belong to this customer"
        if order.status != "pending":
            return f"Order cannot be edited. Current status: {order.status}"

        # Find the product
        product = None
        if str(product_identifier).isdigit():
            product = db.query(Product).filter(Product.id == int(product_identifier), Product.seller_id == order.seller_id).first()
        else:
            product = db.query(Product).filter(Product.name.ilike(f"%{product_identifier}%"), Product.seller_id == order.seller_id).first()
        
        if not product:
            return f"Product '{product_identifier}' not found for this seller"

        # Find the order item
        order_item = db.query(OrderItem).filter(OrderItem.order_id == order.id, OrderItem.product_id == product.id).first()
        
        if not order_item:
            return f"Product '{product.name}' is not in this order"

        # Restock the product and remove from order
        product.stock += order_item.quantity
        order.total_amount -= order_item.price * order_item.quantity
        
        db.delete(order_item)
        db.commit()
        
        return f"Removed '{product.name}' from order. Order total: ${order.total_amount:.2f}"

    except Exception as e:
        db.rollback()
        return f"Error removing item from order: {str(e)}"
    finally:
        db.close()

def update_item_quantity_in_order(customer_id: str, order_id: str, product_identifier: str, new_quantity: int) -> str:
    """
    Update the quantity of a specific item in an existing pending order.
    product_identifier can be product ID (numeric) or product name (string)
    """
    db = SessionLocal()
    try:
        # Validate order exists and belongs to customer
        order = db.query(Order).filter(Order.id == int(order_id), Order.customer_id == customer_id).first()
        if not order:
            return "Order not found or doesn't belong to this customer"
        if order.status != "pending":
            return f"Order cannot be edited. Current status: {order.status}"

        # Find the product
        product = None
        if str(product_identifier).isdigit():
            product = db.query(Product).filter(Product.id == int(product_identifier), Product.seller_id == order.seller_id).first()
        else:
            product = db.query(Product).filter(Product.name.ilike(f"%{product_identifier}%"), Product.seller_id == order.seller_id).first()
        
        if not product:
            return f"Product '{product_identifier}' not found for this seller"

        # Find the order item
        order_item = db.query(OrderItem).filter(OrderItem.order_id == order.id, OrderItem.product_id == product.id).first()
        
        if not order_item:
            return f"Product '{product.name}' is not in this order"

        if new_quantity <= 0:
            return "Quantity must be greater than 0. Use remove_item_from_order to remove items completely"

        # Calculate stock adjustment
        old_quantity = order_item.quantity
        quantity_diff = new_quantity - old_quantity
        
        if quantity_diff > 0 and product.stock < quantity_diff:
            return f"Insufficient stock for product '{product.name}'. Available: {product.stock}, Additional needed: {quantity_diff}"

        # Update quantity and stock
        order_item.quantity = new_quantity
        product.stock -= quantity_diff  # If negative, it adds stock back
        order.total_amount += product.price * quantity_diff
        
        db.commit()
        return f"Updated '{product.name}' quantity from {old_quantity} to {new_quantity}. Order total: ${order.total_amount:.2f}"

    except Exception as e:
        db.rollback()
        return f"Error updating item quantity: {str(e)}"
    finally:
        db.close()

def replace_order_items(customer_id: str, order_id: str, new_items: List[dict]) -> str:
    """
    Replace all items in an existing pending order with new items.
    Each item in new_items must have:
    - 'product_id' (can be numeric ID or product name)
    - 'quantity'
    """
    db = SessionLocal()
    try:
        # Validate order exists and belongs to customer
        order = db.query(Order).filter(Order.id == int(order_id), Order.customer_id == customer_id).first()
        if not order:
            return "Order not found or doesn't belong to this customer"
        if order.status != "pending":
            return f"Order cannot be edited. Current status: {order.status}"

        # First, restock all existing items
        existing_items = db.query(OrderItem).filter(OrderItem.order_id == order.id).all()
        for item in existing_items:
            product = db.query(Product).filter(Product.id == item.product_id).first()
            if product:
                product.stock += item.quantity  # Restock
            db.delete(item)  # Remove old items

        db.flush()

        # Now add new items
        total_amount = 0
        for item in new_items:
            product = None
            identifier = item["product_id"]
            
            if str(identifier).isdigit():
                product = db.query(Product).filter(Product.id == int(identifier), Product.seller_id == order.seller_id).first()
            else:
                product = db.query(Product).filter(Product.name.ilike(f"%{identifier}%"), Product.seller_id == order.seller_id).first()
            
            if not product:
                db.rollback()
                return f"Product '{identifier}' not found for this seller"
            if product.stock < item["quantity"]:
                db.rollback()
                return f"Insufficient stock for product '{product.name}'. Available: {product.stock}, Requested: {item['quantity']}"

            total_amount += product.price * item["quantity"]
            order_item = OrderItem(order_id=order.id, product_id=product.id, price=product.price, quantity=item["quantity"])
            db.add(order_item)
            product.stock -= item["quantity"]

        # Update the total and commit
        order.total_amount = total_amount
        db.commit()

        return f"Order {order.id} successfully updated with {len(new_items)} items. New total: ${total_amount:.2f}"

    except Exception as e:
        db.rollback()
        return f"Error replacing order items: {str(e)}"
    finally:
        db.close()

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
            WHERE o."customer_id" = :customer_id
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
            WHERE o."customer_id" = :customer_id
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
                o."customer_id",
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
            "customer_id": order_info["customer_id"],
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
        product = db.query(Product).filter(Product.id == product_id).first()
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
        order = db.query(Order).filter(Order.id == order_id).first()

        if not order:
            return {"success": False, "error": "Order not found"}
        if order.customer_id != customer_id:
            return {"success": False, "error": "Order does not belong to this customer"}
        if order.status != "pending":
            return {"success": False, "error": "Only pending orders can be edited"}

        # Validate stock for all items first
        for item in new_items:
            product = db.query(Product).filter(Product.id == item["product_id"]).first()
            if not product:
                return {"success": False, "error": f"Product {item['product_id']} not found"}
            if product.stock < item["quantity"]:
                return {
                    "success": False,
                    "error": f"Insufficient stock for product {product.name}"
                }

        # Delete old order items
        db.query(OrderItem).filter(OrderItem.order_id == order_id).delete()

        total = 0
        for item in new_items:
            product = db.query(Product).filter(Product.id == item["product_id"]).first()

            # Create new order item
            order_item = OrderItem(
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

def get_seller_id_by_whatsapp_number_id(whatsapp_number_id: str) -> str:
    """Get seller ID based on WhatsApp number"""
    db = SessionLocal()
    try:
        # Assuming you have a mapping of WhatsApp numbers to seller IDs
        # This is a placeholder logic, replace with actual implementation
        seller = db.query(SellerProfile).filter(SellerProfile.whatsapp_number_id == whatsapp_number_id).first()
        if seller:
            return str(seller.id)
        return "default_seller"  # Fallback if no specific seller found
    finally:
        db.close()



def get_product_images(product_id: int) -> str:
    """Get image URL for a product by its ID"""
    db = SessionLocal()
    try:
        images = db.query(ProductImage).filter(ProductImage.product_id == product_id).all()
        urls = [img.image for img in images if img.image]
        return urls if urls else ["https://www.shutterstock.com/image-photo/person-using-smartphone-interact-friendly-600nw-2482428287.jpg"]
    finally:
        db.close()


# Payment Proof Functions
def upload_payment_proof_for_order(order_id: int, file_name: str, file_type: str, file_content: bytes) -> str:
    """
    Upload payment proof image to S3, save URL to order, then delete the image.
    
    Args:
        order_id: ID of the order to update
        file_name: Name of the payment proof file
        file_type: MIME type of the file
        file_content: File content as bytes
        
    Returns:
        str: Success/error message
    """
    try:
        # Run the async function
        result = asyncio.run(upload_payment_proof(
            order_id=order_id,
            file_name=file_name,
            file_type=file_type,
            file_size=len(file_content),
            file_content=file_content
        ))
        
        if result.get('success'):
            return f"Payment proof uploaded successfully for order {order_id}. File URL: {result.get('file_url')}"
        else:
            return f"Error uploading payment proof: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error uploading payment proof: {str(e)}"

def get_presigned_upload_url(file_name: str, file_type: str, file_size: int, folder: str = "payment-proofs") -> dict:
    """
    Generate a presigned URL for uploading payment proof files.
    
    Args:
        file_name: Name of the file
        file_type: MIME type of the file
        file_size: Size of the file in bytes
        folder: S3 folder to upload to
        
    Returns:
        dict: Contains upload_url and file_url, or error
    """
    try:
        # Run the async function
        result = asyncio.run(generate_presigned_url(
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            folder=folder
        ))
        
        return result
        
    except Exception as e:
        return {"error": f"Error generating presigned URL: {str(e)}"}

def update_order_payment_proof(order_id: int, payment_proof_url: str) -> str:
    """
    Update order with payment proof URL directly.
    
    Args:
        order_id: ID of the order to update
        payment_proof_url: URL of the uploaded payment proof
        
    Returns:
        str: Success/error message
    """
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return f"Order with ID {order_id} not found"
        
        order.payment_proof = payment_proof_url
        db.commit()
        
        return f"Payment proof URL updated successfully for order {order_id}"
        
    except Exception as e:
        db.rollback()
        return f"Error updating payment proof: {str(e)}"
    finally:
        db.close()

def get_order_payment_proof(order_id: int) -> str:
    """
    Get payment proof URL for an order.
    
    Args:
        order_id: ID of the order
        
    Returns:
        str: Payment proof URL or error message
    """
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return f"Order with ID {order_id} not found"
        
        if order.payment_proof:
            return order.payment_proof
        else:
            return f"No payment proof found for order {order_id}"
            
    except Exception as e:
        return f"Error retrieving payment proof: {str(e)}"
    finally:
        db.close()


def upload_payment_proof_and_update_order(order_id: int, file_path: str) -> str:
    """
    Upload payment proof image from local file path to S3, save URL to order database, and delete local file.
    
    Args:
        order_id: ID of the order to update with payment proof
        file_path: Local file path to the payment proof image
        
    Returns:
        str: Success/error message
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File not found at path '{file_path}'"
        
        # Get file information
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Determine file type based on extension
        file_extension = os.path.splitext(file_name)[1].lower()
        file_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.pdf': 'application/pdf'
        }
        
        file_type = file_type_map.get(file_extension)
        if not file_type:
            return f"Error: Unsupported file type '{file_extension}'. Supported types: {list(file_type_map.keys())}"
        
        # Read file content
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
        
        # Check if order exists
        db = SessionLocal()
        try:
            order = db.query(Order).filter(Order.id == order_id).first()
            if not order:
                return f"Error: Order with ID {order_id} not found"
        finally:
            db.close()
        
        # Upload to S3 and update order
        try:
            # Import S3 service locally to avoid import errors if boto3 not installed
            from services.s3_service import s3_service
            
            # Upload file directly to S3
            result = asyncio.run(s3_service.upload_file_direct(
                file_name=file_name,
                file_type=file_type,
                file_content=file_content,
                folder="payment-proofs"
            ))
            
            file_url = result['file_url']
            
            # Update order with payment proof URL
            db = SessionLocal()
            try:
                order = db.query(Order).filter(Order.id == order_id).first()
                order.payment_proof = file_url
                db.commit()
                
                # Delete local file after successful upload and DB update
                try:
                    os.remove(file_path)
                    file_deleted_msg = f" Local file '{file_path}' has been deleted."
                except OSError as e:
                    file_deleted_msg = f" Warning: Could not delete local file '{file_path}': {str(e)}"
                
                return f"✅ Payment proof uploaded successfully for order {order_id}. File URL: {file_url}.{file_deleted_msg}"
                
            except Exception as e:
                db.rollback()
                # If DB update fails, try to delete the S3 file to avoid orphaned files
                try:
                    asyncio.run(s3_service.delete_file(result['key']))
                except:
                    pass  # Ignore S3 cleanup errors
                return f"Error updating order in database: {str(e)}"
            finally:
                db.close()
                
        except ImportError:
            return "Error: S3 service not available - boto3 not installed. Please install boto3: pip install boto3"
        except Exception as e:
            return f"Error uploading to S3: {str(e)}"
            
    except Exception as e:
        return f"Error processing payment proof upload: {str(e)}"


def get_order_payment_status(order_id: int) -> str:
    """
    Get payment proof status and URL for an order.
    
    Args:
        order_id: ID of the order to check
        
    Returns:
        str: Payment proof status and URL or error message
    """
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return f"Order with ID {order_id} not found"
        
        if order.payment_proof:
            return f"Order {order_id} has payment proof uploaded. URL: {order.payment_proof}"
        else:
            return f"Order {order_id} does not have payment proof uploaded yet"
            
    except Exception as e:
        return f"Error retrieving payment status: {str(e)}"
    finally:
        db.close()


def remove_order_payment_proof(order_id: int) -> str:
    """
    Remove payment proof URL from order and optionally delete S3 file.
    
    Args:
        order_id: ID of the order to remove payment proof from
        
    Returns:
        str: Success/error message
    """
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return f"Order with ID {order_id} not found"
        
        if not order.payment_proof:
            return f"Order {order_id} does not have payment proof to remove"
        
        old_url = order.payment_proof
        
        # Try to delete from S3 if possible
        try:
            from services.s3_service import s3_service
            # Extract S3 key from URL
            if 's3.amazonaws.com/' in old_url:
                s3_key = old_url.split('s3.amazonaws.com/')[-1]
                asyncio.run(s3_service.delete_file(s3_key))
                s3_delete_msg = " S3 file has been deleted."
            else:
                s3_delete_msg = " Could not delete S3 file (invalid URL format)."
        except:
            s3_delete_msg = " Could not delete S3 file."
        
        # Remove URL from database
        order.payment_proof = None
        db.commit()
        
        return f"✅ Payment proof removed from order {order_id}.{s3_delete_msg}"
        
    except Exception as e:
        db.rollback()
        return f"Error removing payment proof: {str(e)}"
    finally:
        db.close()
