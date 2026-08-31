from db.database import SessionLocal, read_session
from models.schemas import Product, Order, ChatLog, OrderItem, Customers, SellerProfile, ProductImage
import os
import re
import random
import json
from datetime import datetime
from typing import List, Union, Dict, Any, Optional
from sqlalchemy import text
from collections import defaultdict
from utils.async_bridge import run_sync
from utils.cache import get_cache
from utils.logger import get_logger

logger = get_logger(__name__)

# The WhatsApp number -> seller mapping is read on EVERY inbound message but
# changes about never, so it is cached. Product listings get a short TTL: stale
# stock in a listing is harmless because place_order re-checks stock inside the
# order transaction, and it is invalidated whenever we change stock ourselves.
_seller_cache = get_cache("seller_by_whatsapp", maxsize=256, ttl=300)
_product_cache = get_cache("product_lookups", maxsize=512, ttl=30)


DEFAULT_PRODUCT_IMAGE = (
    "https://www.shutterstock.com/image-photo/"
    "person-using-smartphone-interact-friendly-600nw-2482428287.jpg"
)


def invalidate_product_cache(seller_id: Optional[str] = None) -> None:
    """Drop cached product data after anything that changes stock or price."""
    if seller_id is None:
        _product_cache.clear()
    else:
        _product_cache.delete(f"all:{seller_id}")

# Import storage service (S3 or local filesystem, picked via STORAGE_BACKEND) for payment proof handling
try:
    from services.storage_service import upload_payment_proof, generate_presigned_url
except ImportError:
    # Fallback if boto3 is not installed and STORAGE_BACKEND=s3
    async def upload_payment_proof(*args, **kwargs):
        return {"error": "Storage service not available - boto3 not installed"}

    async def generate_presigned_url(*args, **kwargs):
        return {"error": "Storage service not available - boto3 not installed"}

# LangChain Tools
def get_product_info(product_name: str, seller_id: str) -> Dict[str, Any]:
    """Look up one product by name (or numeric id) for a seller.

    Images are fetched in the same session as the product. This used to call
    get_product_images(), which opened a second connection while the first was
    still checked out - two pool slots per lookup, and a deadlock risk once the
    pool was saturated.
    """
    identifier = str(product_name).strip()
    with read_session() as db:
        query = db.query(Product).filter(Product.seller_id == int(seller_id))
        if identifier.isdigit():
            product = query.filter(Product.id == int(identifier)).first()
        else:
            product = query.filter(Product.name.ilike(f"%{identifier}%")).first()

        if not product:
            return {"error": "Product not found"}

        image_rows = (
            db.query(ProductImage.image_url)
            .filter(ProductImage.product_id == product.id)
            .all()
        )
        images = [url for (url,) in image_rows if url] or [DEFAULT_PRODUCT_IMAGE]

        return {
            "product_id": product.id,
            "product": product.name,
            "description": product.description,
            "price": product.price,
            "stock": product.stock,
            "images": images,
        }

def get_all_products(seller_id: str) -> List[Any]:
    """The seller's catalogue. Cached briefly - it is requested constantly."""

    def _load():
        with read_session() as db:
            rows = (
                db.query(Product.name, Product.price, Product.stock)
                .filter(Product.seller_id == int(seller_id))
                .order_by(Product.name)
                .all()
            )
        if rows:
            return [{"name": n, "price": p, "stock": s} for n, p, s in rows]
        return ["No products found for this seller"]

    return _product_cache.get_or_set(f"all:{seller_id}", _load)

def track_order_detailed(order_id: str) -> Dict[str, Any]:
    """Order status as structured data, so message templates can render it.

    The string-returning track_order() below is what the LLM tool calls; this is
    what the outbound formatter reads.
    """
    try:
        numeric_id = int(str(order_id).strip())
    except (TypeError, ValueError):
        return {"found": False, "error": f"'{order_id}' is not a valid order number"}

    with read_session() as db:
        row = (
            db.query(Order.id, Order.status, Order.created_at, Order.total_amount)
            .filter(Order.id == numeric_id)
            .first()
        )
    if not row:
        return {"found": False, "error": "Order not found"}
    return {
        "found": True,
        "order_id": row.id,
        "status": row.status,
        "created_at": str(row.created_at),
        "total_amount": row.total_amount,
    }

def track_order(order_id: str) -> str:
    result = track_order_detailed(order_id)
    if not result["found"]:
        return result["error"]
    return (
        f"Order ID: {result['order_id']}, Status: {result['status']}, "
        f"Created: {result['created_at']}"
    )

def place_order_detailed(seller_id: str, user_id: str, items: List[dict]) -> Dict[str, Any]:
    """Place an order and return structured details for the templates."""
    message = place_order(seller_id, user_id, items)

    # place_order returns prose; pull the ids back out rather than duplicating
    # the whole transaction here.
    match = re.search(r"Order ID:\s*(\d+).*?Rs\.([\d.]+)", message)
    if match:
        return {
            "success": True,
            "order_id": int(match.group(1)),
            "total_amount": float(match.group(2)),
            "message": message,
            "items": items,
        }
    return {"success": False, "message": message, "error": message, "items": items}

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
        invalidate_product_cache(seller_id)
        return f"Order placed successfully. Order ID: {order.id}, Total Amount: Rs.{total_amount:.2f}"
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
    random_number = random.randint(1000, 9999)
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
        invalidate_product_cache(str(order.seller_id))
        return f"{action}. Order total: Rs.{order.total_amount:.2f}"

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
        invalidate_product_cache(str(order.seller_id))

        return f"Removed '{product.name}' from order. Order total: Rs.{order.total_amount:.2f}"

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
        invalidate_product_cache(str(order.seller_id))
        return f"Updated '{product.name}' quantity from {old_quantity} to {new_quantity}. Order total: Rs.{order.total_amount:.2f}"

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
        invalidate_product_cache(str(order.seller_id))

        return f"Order {order.id} successfully updated with {len(new_items)} items. New total: Rs.{total_amount:.2f}"

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
    """Map an inbound WhatsApp number id to a seller id.

    Runs on every single inbound message, so the result is cached for 5 minutes.
    """

    def _load() -> str:
        with read_session() as db:
            row = (
                db.query(SellerProfile.id)
                .filter(SellerProfile.whatsapp_number_id == whatsapp_number_id)
                .first()
            )
        if row:
            return str(row.id)
        logger.warning(
            "No seller profile mapped to WhatsApp number id %s - falling back to "
            "'default_seller', which will not match any products",
            whatsapp_number_id,
        )
        return "default_seller"

    return _seller_cache.get_or_set(str(whatsapp_number_id), _load)



def get_shop_name(seller_id: str) -> str:
    """The seller's shop name, for use in message templates.

    Cached with the seller lookup TTL - it is read on every outbound message and
    changes about never.
    """

    def _load() -> str:
        try:
            with read_session() as db:
                row = (
                    db.query(SellerProfile.shop_name)
                    .filter(SellerProfile.id == int(seller_id))
                    .first()
                )
            if row and row[0]:
                return str(row[0])
        except (TypeError, ValueError):
            pass  # seller_id isn't numeric (e.g. "default_seller")
        except Exception as e:
            logger.debug("Could not read shop name for seller %s: %s", seller_id, e)
        return "our shop"

    return _seller_cache.get_or_set(f"shop_name:{seller_id}", _load)


def get_product_images(product_id: int) -> List[str]:
    """Image URLs for a product, or a single placeholder if it has none."""
    with read_session() as db:
        rows = (
            db.query(ProductImage.image_url)
            .filter(ProductImage.product_id == product_id)
            .all()
        )
    urls = [url for (url,) in rows if url]
    return urls or [DEFAULT_PRODUCT_IMAGE]


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
        result = run_sync(upload_payment_proof(
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
        result = run_sync(generate_presigned_url(
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
        
        # Upload via configured storage backend (S3 or local) and update order
        try:
            # Import locally to avoid import errors if boto3 not installed (S3 backend only)
            from services.storage_service import storage_service as s3_service

            # Upload file via storage backend
            result = run_sync(s3_service.upload_file_direct(
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
                # If DB update fails, try to delete the uploaded file to avoid orphaned files
                try:
                    run_sync(s3_service.delete_file(result['key']))
                except:
                    pass  # Ignore cleanup errors
                return f"Error updating order in database: {str(e)}"
            finally:
                db.close()

        except ImportError:
            return "Error: S3 service not available - boto3 not installed. Please install boto3: pip install boto3"
        except Exception as e:
            return f"Error uploading payment proof: {str(e)}"
            
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
        
        # Try to delete the stored file if possible (S3 or local, depending on STORAGE_BACKEND)
        try:
            from services.storage_service import storage_service
            from config.storage import S3_ENABLED, LOCAL_STORAGE_BASE_URL

            if S3_ENABLED and 's3.amazonaws.com/' in old_url:
                file_key = old_url.split('s3.amazonaws.com/')[-1]
                run_sync(storage_service.delete_file(file_key))
                s3_delete_msg = " Stored file has been deleted."
            elif not S3_ENABLED and old_url.startswith(LOCAL_STORAGE_BASE_URL + "/"):
                file_key = old_url[len(LOCAL_STORAGE_BASE_URL) + 1:]
                run_sync(storage_service.delete_file(file_key))
                s3_delete_msg = " Stored file has been deleted."
            else:
                s3_delete_msg = " Could not delete stored file (invalid URL format)."
        except:
            s3_delete_msg = " Could not delete stored file."
        
        # Remove URL from database
        order.payment_proof = None
        db.commit()
        
        return f"✅ Payment proof removed from order {order_id}.{s3_delete_msg}"
        
    except Exception as e:
        db.rollback()
        return f"Error removing payment proof: {str(e)}"
    finally:
        db.close()

def cancel_order(customer_id: str, order_id: str, reason: str = "") -> str:
    """
    Cancel an order by updating its status to 'cancelled'.
    Only pending orders can be cancelled.
    Restores product stock when order is cancelled.
    """
    db = SessionLocal()
    try:
        # Find the order
        order = db.query(Order).filter(
            Order.id == order_id,
            Order.customer_id == customer_id
        ).first()
        
        if not order:
            return f"Order {order_id} not found or doesn't belong to this customer."
        
        # Check if order can be cancelled
        if order.status == "cancelled":
            return f"Order {order_id} is already cancelled."
        
        if order.status not in ["pending"]:
            return f"Cannot cancel order {order_id}. Current status: {order.status}. Only pending orders can be cancelled."
        
        # Get order items to restore stock
        order_items = db.query(OrderItem).filter(OrderItem.order_id == order_id).all()
        
        # Restore stock for all items in the order
        for item in order_items:
            product = db.query(Product).filter(Product.id == item.product_id).first()
            if product:
                product.stock += item.quantity
        
        # Update order status to cancelled
        order.status = "cancelled"
        
        # Add cancellation reason if provided
        if reason:
            pass
        
        db.commit()
        invalidate_product_cache(str(order.seller_id))

        return f"Order {order_id} has been successfully cancelled. Stock has been restored for all items."
        
    except Exception as e:
        db.rollback()
        return f"Error cancelling order {order_id}: {str(e)}"
    finally:
        db.close()


# =====================================================================
# Image-aware tools: receipt verification and visual product search
# =====================================================================

# How far the receipt amount may differ from the order total before we flag it.
PAYMENT_TOLERANCE = float(os.getenv("PAYMENT_AMOUNT_TOLERANCE", "1.0"))


def _read_image_file(file_path: str) -> Dict[str, Any]:
    """Shared checks for an image path handed to us by the agent."""
    if not file_path or not os.path.exists(file_path):
        return {"ok": False, "error": f"File not found at path '{file_path}'"}

    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()
    type_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                '.gif': 'image/gif', '.webp': 'image/webp'}
    file_type = type_map.get(ext)
    if not file_type:
        return {"ok": False, "error": f"Unsupported file type '{ext}'. Use JPG, PNG, GIF or WEBP."}

    with open(file_path, 'rb') as f:
        content = f.read()

    return {"ok": True, "file_name": file_name, "file_type": file_type, "content": content}


def verify_and_save_payment_proof(order_id: int, file_path: str, customer_id: str = None) -> str:
    """
    Look at a payment receipt image, check it against the order, then store it.

    Reads the amount/reference/bank off the receipt with the vision model and
    compares the amount to the order total. The proof is always saved (so nothing
    the customer sent is lost), but the order is flagged for human review when the
    image isn't a receipt, can't be read, or the amount doesn't match.

    Returns a human-readable message for the customer.
    """
    db = SessionLocal()
    try:
        try:
            order_id = int(order_id)
        except (TypeError, ValueError):
            return f"Error: '{order_id}' is not a valid order ID."

        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return f"Order with ID {order_id} not found."

        # Make sure this order actually belongs to the person in the chat.
        if customer_id and str(order.customer_id) != str(customer_id):
            logger.warning(
                f"Customer {customer_id} tried to attach payment proof to order {order_id} "
                f"which belongs to {order.customer_id}"
            )
            return f"Order {order_id} does not belong to you. Please check the order number."

        file_info = _read_image_file(file_path)
        if not file_info["ok"]:
            return f"Error: {file_info['error']}"

        # ---- Step 1: read the receipt -------------------------------------
        try:
            from services.image_analysis_service import vision_service
            extraction = vision_service.extract_receipt_details(file_path)
        except Exception as e:
            logger.error(f"Vision check failed for order {order_id}: {str(e)}")
            extraction = {"success": False, "error": str(e)}

        verification = "unreadable"
        flagged = True
        flag_reason = "Could not read the receipt"
        customer_msg = None

        if not extraction.get("success"):
            flag_reason = f"Vision check failed: {extraction.get('error', 'unknown error')}"
            customer_msg = (
                "I've saved your payment slip, but I couldn't read it automatically. "
                "Our team will check it manually and confirm shortly."
            )

        elif not extraction.get("is_receipt"):
            verification = "not_a_receipt"
            flag_reason = "Image does not appear to be a payment receipt"
            customer_msg = (
                "That image doesn't look like a payment receipt. "
                "Please send a clear photo or screenshot of your bank transfer slip."
            )

        else:
            amount = extraction.get("amount")
            issues = extraction.get("issues") or []

            if amount is None:
                verification = "unreadable"
                flag_reason = "Amount could not be read from the receipt"
                if issues:
                    flag_reason += f" ({'; '.join(str(i) for i in issues)})"
                customer_msg = (
                    "I've saved your receipt, but I couldn't clearly read the amount. "
                    "Our team will verify it manually and confirm shortly."
                )

            elif abs(float(amount) - float(order.total_amount)) <= PAYMENT_TOLERANCE:
                verification = "verified"
                flagged = False
                flag_reason = None
                customer_msg = (
                    f"Payment verified. We received your payment of {amount:.2f} "
                    f"for order {order_id}. Thank you!"
                )

            else:
                verification = "amount_mismatch"
                flag_reason = (
                    f"Receipt shows {amount:.2f} but order total is {float(order.total_amount):.2f}"
                )
                customer_msg = (
                    f"I've saved your receipt, but the amount on it ({amount:.2f}) doesn't match "
                    f"the order total ({float(order.total_amount):.2f}). "
                    "Our team will review this and get back to you."
                )

        # ---- Step 2: store the file ---------------------------------------
        try:
            from services.storage_service import storage_service
            upload = run_sync(storage_service.upload_file_direct(
                file_name=file_info["file_name"],
                file_type=file_info["file_type"],
                file_content=file_info["content"],
                folder="payment-proofs",
            ))
            file_url = upload["file_url"]
        except Exception as e:
            logger.error(f"Failed to store payment proof for order {order_id}: {str(e)}")
            return f"Error saving payment proof: {str(e)}"

        # ---- Step 3: write it all to the order -----------------------------
        try:
            order.payment_proof = file_url
            order.payment_verification = verification
            order.payment_flagged = flagged
            order.payment_flag_reason = flag_reason
            order.payment_verified_at = datetime.utcnow()

            if extraction.get("success"):
                order.payment_amount = extraction.get("amount")
                order.payment_currency = extraction.get("currency")
                order.payment_reference = extraction.get("reference")
                order.payment_bank = extraction.get("bank")
                order.payment_date = extraction.get("date")
                order.payment_raw_extraction = {
                    k: v for k, v in extraction.items() if k != "success"
                }

            # Keep the Admin Portal's own field in its expected vocabulary.
            order.payment_status = "Paid" if verification == "verified" else "Pending"

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"DB update failed for order {order_id}: {str(e)}")
            # Don't leave an orphaned file behind
            try:
                run_sync(storage_service.delete_file(upload["key"]))
            except Exception:
                pass
            return f"Error updating order {order_id}: {str(e)}"

        # Local download is no longer needed
        try:
            os.remove(file_path)
        except OSError:
            pass

        logger.info(
            f"Payment proof for order {order_id}: {verification} "
            f"(flagged={flagged}, reason={flag_reason})"
        )
        return customer_msg

    except Exception as e:
        db.rollback()
        logger.error(f"verify_and_save_payment_proof failed: {str(e)}")
        return f"Error processing payment proof: {str(e)}"
    finally:
        db.close()


def verify_and_save_payment_proof_detailed(
    order_id: int, file_path: str, customer_id: str = None
) -> Dict[str, Any]:
    """verify_and_save_payment_proof plus the verification fields it wrote.

    The outbound templates need to know whether the receipt matched, and by how
    much. Rather than thread a second return value through that whole function,
    this reads the columns it just committed - one indexed lookup, and only on a
    receipt image, which is rare.
    """
    message = verify_and_save_payment_proof(order_id, file_path, customer_id)
    details: Dict[str, Any] = {"message": message, "order_id": order_id}

    try:
        with read_session() as db:
            row = (
                db.query(
                    Order.payment_verification,
                    Order.payment_amount,
                    Order.payment_currency,
                    Order.payment_reference,
                    Order.payment_flagged,
                    Order.payment_flag_reason,
                    Order.total_amount,
                )
                .filter(Order.id == int(order_id))
                .first()
            )
        if row:
            details.update(
                {
                    "verification": row.payment_verification,
                    "amount": row.payment_amount,
                    "currency": row.payment_currency,
                    "reference": row.payment_reference,
                    "flagged": row.payment_flagged,
                    "flag_reason": row.payment_flag_reason,
                    "total_amount": row.total_amount,
                }
            )
    except Exception as e:
        # The proof is already saved; failing to read it back is cosmetic.
        logger.debug("Could not read back payment verification for %s: %s", order_id, e)

    return details


def find_similar_products_by_image(file_path: str, seller_id: str) -> str:
    """
    Customer sent a photo of something - find what we sell that matches.

    The vision model describes the item, then we match that description against
    the seller's catalogue by name and description keywords.
    """
    file_info = _read_image_file(file_path)
    if not file_info["ok"]:
        return f"Error: {file_info['error']}"

    # ---- Step 1: describe what's in the photo -----------------------------
    try:
        from services.image_analysis_service import vision_service
        described = vision_service.describe_product_image(file_path)
    except Exception as e:
        logger.error(f"Vision description failed: {str(e)}")
        return "I couldn't analyse that image. Could you tell me what you're looking for?"

    if not described.get("success"):
        return (
            "I couldn't make out what's in that image. "
            "Could you describe what you're looking for?"
        )

    item_type = described.get("item_type") or ""
    keywords = [str(k) for k in (described.get("keywords") or [])]
    colors = [str(c) for c in (described.get("colors") or [])]
    description = described.get("description", "")

    # ---- Step 2: match against the catalogue ------------------------------
    db = SessionLocal()
    try:
        products = db.query(Product).filter(Product.seller_id == int(seller_id)).all()
        if not products:
            return "We don't have any products listed yet."

        # Score each product by how many of the vision terms appear in its
        # name or description. Item type counts double - it's the strongest signal.
        terms = [t.lower() for t in ([item_type] + keywords + colors) if t]
        scored = []

        for p in products:
            haystack = f"{p.name or ''} {p.description or ''}".lower()
            score = 0
            for term in terms:
                # Word-boundary match so "tea" doesn't match "steam"
                if re.search(r'\b' + re.escape(term) + r'\b', haystack):
                    score += 2 if term == item_type.lower() else 1
            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: (-x[0], x[1].name or ""))
        matches = [p for _, p in scored[:5]]

        if not matches:
            return (
                f"I can see this is {description or 'an item'}, but I couldn't find "
                f"anything matching it in our catalogue. "
                "Would you like to see what we do have?"
            )

        lines = [f"I can see this is {description}", "", "Here's what we have that's similar:"]
        for p in matches:
            stock_note = f"{p.stock} in stock" if p.stock and p.stock > 0 else "out of stock"
            lines.append(f"- {p.name} - {p.price:.2f} ({stock_note})")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Product image search failed: {str(e)}")
        return f"Error searching for similar products: {str(e)}"
    finally:
        db.close()


def get_flagged_payment_orders(seller_id: str) -> str:
    """List orders whose payment proof needs a human to look at it."""
    db = SessionLocal()
    try:
        orders = (
            db.query(Order)
            .filter(Order.seller_id == int(seller_id), Order.payment_flagged == True)  # noqa: E712
            .order_by(Order.created_at.desc())
            .all()
        )
        if not orders:
            return "No payment proofs are waiting for review."

        lines = [f"{len(orders)} payment proof(s) need review:"]
        for o in orders:
            lines.append(
                f"- Order {o.id}: {o.payment_verification or 'unknown'} - "
                f"{o.payment_flag_reason or 'no reason recorded'} "
                f"(order total {float(o.total_amount):.2f})"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching flagged orders: {str(e)}"
    finally:
        db.close()
