from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from db.database import Base
from datetime import datetime

# Database Models
class Seller(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    seller_id = Column(Integer, ForeignKey("seller_profiles.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    price = Column(Float, nullable=False)
    stock = Column(Integer, nullable=False)

class Customers(Base):
    __tablename__ = "customers"
    id = Column(String, primary_key=True)  # Assuming user_id is a string
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    address = Column(String, nullable=True)
    number1 = Column(String, nullable=False) 
    number2 = Column(String, nullable=False) 
    created_at = Column(DateTime, default=datetime.utcnow)

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    seller_id = Column(Integer, ForeignKey("seller_profiles.id"), nullable=False)
    customer_id = Column(String,ForeignKey("customers.id"), nullable=False)
    status = Column(String, nullable=False)
    total_amount = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)

class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True)
    seller_id = Column(Integer, ForeignKey("seller_profiles.id"), nullable=False)
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    user_query = Column(String, nullable=False)
    intent = Column(String)
    entities = Column(String)
    response = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class SellerProfile(Base):
    __tablename__ = "seller_profiles"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    shop_name = Column(String, nullable=True)
    gst_number = Column(String, nullable=True)
    whatsapp_number_id = Column(String, nullable=True)
 