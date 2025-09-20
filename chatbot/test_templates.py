#!/usr/bin/env python3
"""
Test script to demonstrate the beautiful message templates with emojis
"""

# Add the current directory to Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.agent import MessageTemplates

def test_product_details_template():
    print("🧪 Testing Product Details Template")
    print("=" * 50)
    
    # Test with string format (as returned by actual get_product_info)
    product_str = "Product ID: 1, Product: Gaming Laptop, Description: High-performance gaming laptop with RTX 4080, Price: Rs.350000, Stock: 15, Images: laptop1.jpg, laptop2.jpg"
    result = MessageTemplates.product_details(product_str)
    print(result)
    print("\n")
    
    # Test with dictionary format
    product_dict = {
        "id": "2",
        "name": "Wireless Gaming Mouse",
        "description": "RGB wireless gaming mouse with 12000 DPI",
        "price": 8500,
        "stock": 25,
        "images": "mouse1.jpg, mouse2.jpg, mouse3.jpg"
    }
    result2 = MessageTemplates.product_details(product_dict)
    print(result2)
    print("\n")

def test_product_list_template():
    print("🧪 Testing Product List Template")
    print("=" * 50)
    
    # Test with list of strings
    products_list = [
        "Gaming Laptop - Rs.350000",
        "Wireless Mouse - Rs.8500", 
        "Mechanical Keyboard - Rs.12000",
        "Gaming Headset - Rs.15000",
        "Monitor 27inch - Rs.45000"
    ]
    result = MessageTemplates.product_list(products_list)
    print(result)
    print("\n")

def test_order_details_template():
    print("🧪 Testing Order Details Template")
    print("=" * 50)
    
    # Test with dictionary format
    order_dict = {
        "id": "12345",
        "customer": "John Doe",
        "total_amount": 378500,
        "status": "confirmed",
        "items": [
            {"name": "Gaming Laptop", "quantity": 1, "price": 350000},
            {"name": "Wireless Mouse", "quantity": 2, "price": 8500},
            {"name": "Mechanical Keyboard", "quantity": 1, "price": 12000}
        ]
    }
    result = MessageTemplates.order_details(order_dict)
    print(result)
    print("\n")

def test_tracking_status_template():
    print("🧪 Testing Tracking Status Template")
    print("=" * 50)
    
    # Test with different statuses
    tracking_data = {
        "order_id": "12345",
        "status": "shipped",
        "estimated_delivery": "2025-09-23"
    }
    result = MessageTemplates.tracking_status(tracking_data)
    print(result)
    print("\n")

def test_customer_info_template():
    print("🧪 Testing Customer Info Template")
    print("=" * 50)
    
    customer_data = {
        "name": "Nimal Perera",
        "email": "nimal@example.com",
        "phone": "+94771234567",
        "address": "123 Galle Road, Colombo 03"
    }
    result = MessageTemplates.customer_info(customer_data)
    print(result)
    print("\n")

def test_payment_confirmation_template():
    print("🧪 Testing Payment Confirmation Template")
    print("=" * 50)
    
    payment_data = {
        "method": "bank_transfer",
        "amount": 378500,
        "order_id": "12345"
    }
    result = MessageTemplates.payment_confirmation(payment_data)
    print(result)
    print("\n")

def test_error_template():
    print("🧪 Testing Error Message Template")
    print("=" * 50)
    
    result1 = MessageTemplates.error_message("not_found", "The product you're looking for is not in our current catalog.")
    print(result1)
    print("\n")
    
    result2 = MessageTemplates.error_message("out_of_stock", "Gaming Laptop is currently out of stock. Expected restock: Next week.")
    print(result2)
    print("\n")

if __name__ == "__main__":
    print("🚀 BEAUTIFUL MESSAGE TEMPLATES DEMO")
    print("=" * 60)
    print()
    
    test_product_details_template()
    test_product_list_template() 
    test_order_details_template()
    test_tracking_status_template()
    test_customer_info_template()
    test_payment_confirmation_template()
    test_error_template()
    
    print("✅ All template tests completed!")
    print("🎨 These beautiful templates will make your chat responses more engaging!")