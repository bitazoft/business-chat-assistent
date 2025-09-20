#!/usr/bin/env python3
"""
Demo script to showcase the beautiful message templates with emojis.
This script demonstrates how the templates work with sample data.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from templates.message_templates import MessageTemplates

def demo_product_details():
    """Demo product details template"""
    print("🔸" * 50)
    print("PRODUCT DETAILS TEMPLATE DEMO")
    print("🔸" * 50)
    
    # Test with string format (as returned by get_product_info)
    product_string = "Product ID: 1, Product: MacBook Pro M3, Description: Latest Apple MacBook with M3 chip, 16GB RAM, 512GB SSD, Price: Rs.450000, Stock: 15, Images: https://example.com/macbook1.jpg, https://example.com/macbook2.jpg"
    
    print("\n1. Product from string format:")
    print(MessageTemplates.product_details(product_string))
    
    # Test with dictionary format
    product_dict = {
        'id': 2,
        'name': 'iPhone 15 Pro',
        'description': 'Latest iPhone with titanium design, A17 Pro chip, and advanced camera system',
        'price': 285000,
        'stock': 25,
        'images': 'https://example.com/iphone1.jpg, https://example.com/iphone2.jpg'
    }
    
    print("\n2. Product from dictionary format:")
    print(MessageTemplates.product_details(product_dict))

def demo_product_list():
    """Demo product list template"""
    print("\n🔸" * 50)
    print("PRODUCT LIST TEMPLATE DEMO")
    print("🔸" * 50)
    
    # Test with list of strings
    products_list = [
        "MacBook Pro M3 - Rs.450000",
        "iPhone 15 Pro - Rs.285000", 
        "Samsung Galaxy S24 - Rs.215000",
        "Dell XPS 13 - Rs.275000",
        "iPad Air M2 - Rs.185000"
    ]
    
    print("\n1. Product list from strings:")
    print(MessageTemplates.product_list(products_list))
    
    # Test with dictionary format
    products_dict = [
        {'name': 'Gaming Chair Pro', 'price': 35000, 'stock': 12},
        {'name': 'Mechanical Keyboard RGB', 'price': 15000, 'stock': 8},
        {'name': 'Wireless Mouse', 'price': 5500, 'stock': 20}
    ]
    
    print("\n2. Product list from dictionaries:")
    print(MessageTemplates.product_list(products_dict))

def demo_order_details():
    """Demo order details template"""
    print("\n🔸" * 50)
    print("ORDER DETAILS TEMPLATE DEMO")
    print("🔸" * 50)
    
    # Test with string format
    order_string = "Order #12345 placed successfully! Total: Rs.150000. Items: MacBook Pro x1, iPhone Case x2. Status: Pending"
    
    print("\n1. Order from string format:")
    print(MessageTemplates.order_details(order_string))
    
    # Test with dictionary format
    order_dict = {
        'id': 12346,
        'customer': 'John Silva',
        'total_amount': 320000,
        'status': 'processing',
        'items': [
            {'name': 'MacBook Pro M3', 'quantity': 1, 'price': 450000},
            {'name': 'iPhone 15', 'quantity': 2, 'price': 185000}
        ]
    }
    
    print("\n2. Order from dictionary format:")
    print(MessageTemplates.order_details(order_dict))

def demo_tracking_status():
    """Demo tracking status template"""
    print("\n🔸" * 50)
    print("ORDER TRACKING TEMPLATE DEMO")
    print("🔸" * 50)
    
    # Test with string format
    tracking_string = "Order #12345 is currently being processed at our warehouse. Expected delivery: 2-3 business days."
    
    print("\n1. Tracking from string format:")
    print(MessageTemplates.tracking_status(tracking_string))
    
    # Test with dictionary format - pending order
    tracking_pending = {
        'order_id': 12347,
        'status': 'pending',
        'estimated_delivery': 'September 25, 2025'
    }
    
    print("\n2. Tracking - Pending Order:")
    print(MessageTemplates.tracking_status(tracking_pending))
    
    # Test with dictionary format - shipped order
    tracking_shipped = {
        'order_id': 12348,
        'status': 'shipped', 
        'estimated_delivery': 'September 22, 2025'
    }
    
    print("\n3. Tracking - Shipped Order:")
    print(MessageTemplates.tracking_status(tracking_shipped))

def demo_customer_info():
    """Demo customer info template"""
    print("\n🔸" * 50)
    print("CUSTOMER INFO TEMPLATE DEMO")
    print("🔸" * 50)
    
    # Test with string format
    customer_string = "Name: Nimal Perera, Email: nimal@gmail.com, Phone: +94771234567, Address: 123/A Galle Road, Colombo 03"
    
    print("\n1. Customer from string format:")
    print(MessageTemplates.customer_info(customer_string))
    
    # Test with dictionary format
    customer_dict = {
        'name': 'Sarah Johnson',
        'email': 'sarah.j@outlook.com',
        'number1': '+94773456789',
        'address': '456/B Kandy Road, Peradeniya'
    }
    
    print("\n2. Customer from dictionary format:")
    print(MessageTemplates.customer_info(customer_dict))

def demo_payment_confirmation():
    """Demo payment confirmation template"""
    print("\n🔸" * 50)
    print("PAYMENT CONFIRMATION TEMPLATE DEMO")
    print("🔸" * 50)
    
    # COD Payment
    payment_cod = {
        'method': 'cod',
        'amount': 150000,
        'order_id': 12349
    }
    
    print("\n1. Cash on Delivery Payment:")
    print(MessageTemplates.payment_confirmation(payment_cod))
    
    # Bank Transfer Payment
    payment_bank = {
        'method': 'bank_transfer',
        'amount': 285000,
        'order_id': 12350
    }
    
    print("\n2. Bank Transfer Payment:")
    print(MessageTemplates.payment_confirmation(payment_bank))

def demo_error_messages():
    """Demo error message template"""
    print("\n🔸" * 50)
    print("ERROR MESSAGE TEMPLATE DEMO")
    print("🔸" * 50)
    
    # Different error types
    print("\n1. Product Not Found:")
    print(MessageTemplates.error_message("not_found", "The product 'MacBook Pro X1' was not found in our inventory"))
    
    print("\n2. Out of Stock:")
    print(MessageTemplates.error_message("out_of_stock", "iPhone 15 Pro is currently out of stock. Expected restock: October 1st"))
    
    print("\n3. Invalid Input:")
    print(MessageTemplates.error_message("invalid_input", "Please provide a valid quantity (positive number)"))
    
    print("\n4. System Error:")
    print(MessageTemplates.error_message("system_error", "Database connection temporarily unavailable. Please try again in a few minutes"))

def demo_success_messages():
    """Demo success message template"""
    print("\n🔸" * 50)
    print("SUCCESS MESSAGE TEMPLATE DEMO")
    print("🔸" * 50)
    
    # Different success types
    print("\n1. Order Placed:")
    print(MessageTemplates.success_message("order_placed", "Your order #12351 has been successfully placed and will be processed within 24 hours"))
    
    print("\n2. Account Created:")
    print(MessageTemplates.success_message("user_created", "Welcome to our store! Your account has been set up and you can now start shopping"))
    
    print("\n3. Profile Updated:")
    print(MessageTemplates.success_message("user_updated", "Your contact information and delivery address have been successfully updated"))

def demo_order_summary():
    """Demo order summary template"""
    print("\n🔸" * 50)
    print("ORDER SUMMARY TEMPLATE DEMO")
    print("🔸" * 50)
    
    # List of orders
    orders_list = [
        {
            'id': 12345,
            'status': 'delivered',
            'total_amount': 450000,
            'created_at': '2025-09-15'
        },
        {
            'id': 12346,
            'status': 'shipped',
            'total_amount': 285000,
            'created_at': '2025-09-18'
        },
        {
            'id': 12347,
            'status': 'pending',
            'total_amount': 75000,
            'created_at': '2025-09-20'
        }
    ]
    
    print("\n1. Order History:")
    print(MessageTemplates.order_summary(orders_list))
    
    print("\n2. Empty Order History:")
    print(MessageTemplates.order_summary([]))

def demo_welcome_message():
    """Demo welcome message template"""
    print("\n🔸" * 50)
    print("WELCOME MESSAGE TEMPLATE DEMO")
    print("🔸" * 50)
    
    print("\n1. General Welcome:")
    print(MessageTemplates.welcome_message())
    
    print("\n2. Electronics Store Welcome:")
    print(MessageTemplates.welcome_message("TechWorld", "electronics"))
    
    print("\n3. Clothing Store Welcome:")
    print(MessageTemplates.welcome_message("Fashion Hub", "clothing"))

def main():
    """Run all template demos"""
    print("=" * 80)
    print("🎨 BEAUTIFUL MESSAGE TEMPLATES DEMO 🎨")
    print("=" * 80)
    print("This demo shows various message templates with emojis for")
    print("a better user experience in business chat applications.")
    print("=" * 80)
    
    try:
        demo_welcome_message()
        demo_product_details()
        demo_product_list()
        demo_order_details()
        demo_tracking_status()
        demo_customer_info()
        demo_payment_confirmation()
        demo_success_messages()
        demo_error_messages()
        demo_order_summary()
        
        print("\n" + "=" * 80)
        print("✅ ALL TEMPLATE DEMOS COMPLETED SUCCESSFULLY!")
        print("The templates are ready to use in your business chat application.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error running demos: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())