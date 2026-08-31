"""
Tool argument schemas.

These used to be declared inside OptimizedChatbot._create_tools(), which meant
every chat request built roughly twenty fresh Pydantic model classes. Building a
model class compiles a validator - it is one of the more expensive things you can
do per request, and the result was identical every time.

Defining them at module level builds them once at import. The examples in each
Field are kept verbatim: they end up in the JSON schema the model sees, so
changing them changes how well the model fills the arguments in.
"""
from typing import List, Union

from pydantic import BaseModel, Field


class GetProductInfoInput(BaseModel):
    product_name: str = Field(
        description="Name or identifier of the product to get information about",
        examples=["laptop", "iPhone 15", "gaming chair", "wireless headphones", "1"],
    )


class TrackOrderInput(BaseModel):
    order_id: str = Field(
        description="Order ID to track - can be numeric string or alphanumeric",
        examples=["12345", "ORD001", "67890", "ORDER_ABC123"],
    )


class PlaceOrderInput(BaseModel):
    items: List[dict] = Field(
        description=(
            "List of items to order. Each item should be a dictionary with 'product_id' (int) "
            "and 'quantity' (int) keys. Example: [{'product_id': 1, 'quantity': 2}, "
            "{'product_id': 3, 'quantity': 1}] use get productInfo method to get the product ids"
        ),
        examples=[
            [{"product_id": 1, "quantity": 2}],
            [{"product_id": 3, "quantity": 1}, {"product_id": 2, "quantity": 3}],
        ],
    )


class SaveUserInput(BaseModel):
    name: str = Field(
        description="User's full name (first and last name)",
        examples=["John Smith", "Nimal Perera", "Sarah Johnson"],
    )
    email: str = Field(
        description="User's email address in valid email format",
        examples=["john.smith@gmail.com", "nimal@yahoo.com", "sarah.j@outlook.com"],
    )
    address: str = Field(
        description="User's complete physical address including city/area",
        examples=["123 Main St, Colombo 07", "45/2 Galle Road, Mount Lavinia", "78 Kandy Road, Peradeniya"],
    )
    number: str = Field(
        description="User's phone number (mobile or landline)",
        examples=["+94771234567", "0771234567", "011-2345678", "+94112345678"],
    )


class UpdateUserInfoInput(BaseModel):
    name: str = Field(
        description="User's name to update (optional, leave empty if not changing)",
        default="",
        examples=["John Smith", "පීතර සිල්වා", ""],
    )
    email: str = Field(
        description="User's email to update (optional, leave empty if not changing)",
        default="",
        examples=["john.new@gmail.com", "updated@email.com", ""],
    )
    address: str = Field(
        description="User's address to update (optional, leave empty if not changing)",
        default="",
        examples=["New Address, Colombo 05", "456 Updated St, Kandy", ""],
    )
    number: str = Field(
        description="User's phone to update (optional, leave empty if not changing)",
        default="",
        examples=["+94779876543", "0119876543", ""],
    )


class OrderItemInput(BaseModel):
    product_id: Union[int, str] = Field(
        ...,
        description="Product ID (numeric) or product name (string)",
        examples=[1, "laptop", 25, "iPhone 15", "gaming_mouse"],
    )
    quantity: int = Field(
        ...,
        gt=0,
        description="Quantity of the product (must be positive integer)",
        examples=[1, 2, 5, 10, 3],
    )


class EditOrderInput(BaseModel):
    customer_id: str = Field(
        ...,
        description="User ID who placed the order",
        examples=["user123", "customer_456", "USR789"],
    )
    order_id: Union[str, int] = Field(
        ...,
        description="Order ID to be edited (numeric or string)",
        examples=[12345, "ORD001", 67890, "ORDER_ABC"],
    )
    new_items: List[OrderItemInput] = Field(
        ...,
        description="Updated complete list of order items (replaces existing items)",
    )


class AddItemToOrderInput(BaseModel):
    order_id: str = Field(
        ...,
        description="Order ID to add item to",
        examples=["12345", "ORD001", "67890"],
    )
    product_identifier: str = Field(
        ...,
        description="Product ID (numeric) or product name (string)",
        examples=["1", "laptop", "25", "iPhone 15", "gaming_mouse"],
    )
    quantity: int = Field(
        ...,
        gt=0,
        description="Quantity to add (positive integer)",
        examples=[1, 2, 3, 5, 10],
    )


class RemoveItemFromOrderInput(BaseModel):
    order_id: str = Field(
        ...,
        description="Order ID to remove item from",
        examples=["12345", "ORD001", "67890"],
    )
    product_identifier: str = Field(
        ...,
        description="Product ID (numeric) or product name (string) to remove",
        examples=["1", "laptop", "25", "iPhone 15"],
    )


class UpdateItemQuantityInput(BaseModel):
    order_id: str = Field(
        ...,
        description="Order ID to update item quantity in",
        examples=["12345", "ORD001", "67890"],
    )
    product_identifier: str = Field(
        ...,
        description="Product ID (numeric) or product name (string) to update",
        examples=["1", "laptop", "25", "iPhone 15"],
    )
    new_quantity: int = Field(
        ...,
        gt=0,
        description="New quantity for the item (positive integer)",
        examples=[1, 2, 5, 10, 15],
    )


class ReplaceOrderItemsInput(BaseModel):
    order_id: str = Field(
        ...,
        description="Order ID to completely replace items in",
        examples=["12345", "ORD001", "67890"],
    )
    new_items: List[OrderItemInput] = Field(
        ...,
        description="Complete new list of order items (replaces all existing items)",
    )


class GetOrdersInput(BaseModel):
    customer_id: str = Field(
        ...,
        description="User ID to retrieve all orders for",
        examples=["user123", "customer_456", "USR789"],
    )


class GetOrderDetailsInput(BaseModel):
    order_id: Union[int, str] = Field(
        ...,
        description="Specific Order ID to retrieve detailed information for",
        examples=[12345, "ORD001", 67890, "ORDER_ABC123"],
    )


class CheckStockInput(BaseModel):
    product_id: Union[int, str] = Field(
        ...,
        description="Product ID (numeric) or name (string) to check stock for",
        examples=[1, "laptop", 25, "iPhone 15", "gaming_mouse"],
    )
    quantity: int = Field(
        ...,
        gt=0,
        description="Quantity to verify if available in stock",
        examples=[1, 2, 5, 10, 20],
    )


class UploadPaymentProofInput(BaseModel):
    order_id: str = Field(
        ...,
        description="Order ID to upload payment proof for",
        examples=["12345", "ORD001", "67890"],
    )
    payment_proof_file: str = Field(
        ...,
        description="File path to payment proof image (bank transfer receipt, etc.)",
        examples=["/uploads/payment_123.jpg", "payment_receipt.png", "/tmp/bank_transfer.pdf"],
    )


class CancelOrderInput(BaseModel):
    order_id: str = Field(
        ...,
        description="Order ID to cancel (only pending orders can be cancelled)",
        examples=["12345", "ORD001", "67890"],
    )
    reason: str = Field(
        default="",
        description="Optional reason for cancellation",
        examples=["Changed mind", "Found better price", "No longer needed", "Ordered by mistake", ""],
    )


class VerifyPaymentProofInput(BaseModel):
    order_id: str = Field(
        ...,
        description="Order ID the payment receipt belongs to",
        examples=["12345", "7", "8"],
    )
    payment_proof_file: str = Field(
        ...,
        description=(
            "File path of the receipt image the customer sent, exactly as given in the "
            "[Image received: ...] message"
        ),
        examples=["./downloads/whatsapp_image_94771234567_20260829_120301_ab12cd34.jpg"],
    )


class ProductImageSearchInput(BaseModel):
    image_file: str = Field(
        ...,
        description=(
            "File path of the product photo the customer sent, exactly as given in the "
            "[Image received: ...] message"
        ),
        examples=["./downloads/whatsapp_image_94771234567_20260829_120301_ab12cd34.jpg"],
    )


class EscalateToHumanInput(BaseModel):
    reason: str = Field(
        ...,
        description=(
            "Short description of why a human is needed, in English, for the shop staff to read"
        ),
        examples=[
            "Customer is asking for a refund",
            "Customer is unhappy with the delivery and wants to complain",
            "Question about wholesale pricing I cannot answer",
        ],
    )


class EmptyInput(BaseModel):
    """For tools whose arguments all come from the session, not the model."""

    pass
