from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

from db.database import Base

# Note on indexes: index=True here only affects create_all / fresh databases.
# For the existing database the same indexes are applied by
# database/migrations/002_performance_indexes.sql, which is safe to re-run.


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
    # Every product query filters on seller_id; without this it was a full scan.
    seller_id = Column(Integer, ForeignKey("seller_profiles.id"), nullable=False, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(String)
    price = Column(Float, nullable=False)
    stock = Column(Integer, nullable=False)

    __table_args__ = (
        Index("idx_products_seller_name", "seller_id", "name"),
    )

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
    seller_id = Column(Integer, ForeignKey("seller_profiles.id"), nullable=False, index=True)
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False, index=True)
    status = Column(String, nullable=False, index=True)
    total_amount = Column(Float, nullable=False)
    payment_proof = Column(String, nullable=True)  # URL to payment proof image
    created_at = Column(DateTime, default=datetime.utcnow)

    # Existing columns the Admin Portal owns (were missing from this model)
    payment_method = Column(String, nullable=True)   # COD | Bank Transfer
    payment_status = Column(String, nullable=True)   # Paid | Pending | Processing | Failed | Rejected
    shipping_cost = Column(String, nullable=True)
    notes = Column(String, nullable=True)

    # --- Payment proof verification (filled in by the vision check) ---
    # verified | amount_mismatch | not_a_receipt | unreadable
    payment_verification = Column(String, nullable=True)
    payment_amount = Column(Float, nullable=True)       # amount read off the receipt
    payment_currency = Column(String, nullable=True)
    payment_reference = Column(String, nullable=True)   # bank transaction / reference number
    payment_bank = Column(String, nullable=True)
    payment_date = Column(String, nullable=True)        # date printed on the receipt
    payment_flagged = Column(Boolean, default=False)    # true = needs a human to look
    payment_flag_reason = Column(String, nullable=True)
    payment_verified_at = Column(DateTime, nullable=True)
    payment_raw_extraction = Column(JSONB, nullable=True)  # everything the vision model returned

    __table_args__ = (
        # "this customer's pending orders" is the single hottest order query -
        # the payment flow runs it on every receipt image.
        Index("idx_orders_customer_status", "customer_id", "status"),
        Index("idx_orders_seller_created", "seller_id", "created_at"),
    )

class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)

    __table_args__ = (
        Index("idx_order_items_order_product", "order_id", "product_id"),
    )

class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True)
    seller_id = Column(Integer, ForeignKey("seller_profiles.id"), nullable=False, index=True)
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False, index=True)
    user_query = Column(String, nullable=False)
    intent = Column(String)
    entities = Column(JSONB)
    response = Column(String, nullable=False)
    response_time_ms = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("idx_chat_logs_seller_timestamp", "seller_id", "timestamp"),
    )

class SellerProfile(Base):
    __tablename__ = "seller_profiles"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    shop_name = Column(String, nullable=True)
    gst_number = Column(String, nullable=True)
    # Looked up on every single inbound WhatsApp message.
    whatsapp_number_id = Column(String, nullable=True, index=True)

class ProductImage(Base):
    __tablename__ = "item_img"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    image_url = Column(String, nullable=False)


class ConversationMessage(Base):
    """One turn of a conversation, persisted so context survives a restart.

    Chat history used to live only in the in-memory session dict, so a deploy or
    crash mid-order dropped everything the customer had told us and the bot would
    start asking for their details again.
    """
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True)
    seller_id = Column(String, nullable=False)
    customer_id = Column(String, nullable=False)
    role = Column(String, nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        # Loading a session reads the newest N rows for one (seller, customer).
        Index("idx_conv_seller_customer_created", "seller_id", "customer_id", "created_at"),
    )


class TokenUsage(Base):
    """What one conversation turn cost, in tokens and money.

    One row per turn rather than per LLM call: a turn can make several model
    calls (one per tool round trip) and the seller is billed for the turn.
    """
    __tablename__ = "token_usage"

    id = Column(Integer, primary_key=True)
    seller_id = Column(String, nullable=False)
    customer_id = Column(String, nullable=False)
    model = Column(String, nullable=False)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    llm_calls = Column(Integer, nullable=False, default=1)
    # Numeric, not float: these get summed into invoices.
    cost_usd = Column(Numeric(12, 6), nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_token_usage_seller_created", "seller_id", "created_at"),
        Index("idx_token_usage_customer_created", "customer_id", "created_at"),
        Index("idx_token_usage_model", "model"),
    )


class MessageTemplate(Base):
    """A seller-customisable outbound message template.

    seller_id NULL means "the default for every seller"; a row with a seller_id
    overrides that default for that seller only.
    """
    __tablename__ = "message_templates"

    id = Column(Integer, primary_key=True)
    seller_id = Column(String, nullable=True)
    template_key = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    description = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        # One row per (seller, key). The partial-unique index that makes the
        # NULL-seller default unique too lives in the migration, since NULLs
        # don't collide in a plain unique constraint.
        UniqueConstraint("seller_id", "template_key", name="uq_message_templates_seller_key"),
        Index("idx_message_templates_key", "template_key"),
    )
