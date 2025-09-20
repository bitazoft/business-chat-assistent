#!/usr/bin/env python3
"""
Demo script to test the beautiful message templates with emojis
"""
import sys
import os
sys.path.append('/home/nipul/business-chat-assistent/chatbot')

from templates.message_templates import MessageTemplates

def demo_templates():
    """Demonstrate all the beautiful templates"""
    
    print("=" * 60)
    print("🎨 BEAUTIFUL MESSAGE TEMPLATES DEMO")
    print("=" * 60)
    
    # Demo 1: Product Details Template
    print("\n1️⃣  PRODUCT DETAILS TEMPLATE")
    print("-" * 40)
    sample_product = {
        'id': 1,
        'name': 'iPhone 15 Pro Max',
        'description': 'Latest iPhone with advanced camera system and A17 Pro chip',
        'price': 450000,
        'stock': 25,
        'images': 'https://example.com/iphone1.jpg, https://example.com/iphone2.jpg'
    }
    print(MessageTemplates.product_details(sample_product))
    
    # Demo 2: Product List Template
    print("\n\n2️⃣  PRODUCT LIST TEMPLATE")
    print("-" * 40)
    sample_products = [
        {'name': 'iPhone 15 Pro Max', 'price': 450000, 'stock': 25},
        {'name': 'Samsung Galaxy S24', 'price': 380000, 'stock': 15},
        {'name': 'MacBook Pro M3', 'price': 850000, 'stock': 8},
        {'name': 'AirPods Pro 2', 'price': 85000, 'stock': 50},
        {'name': 'iPad Air', 'price': 220000, 'stock': 20}
    ]
    print(MessageTemplates.product_list(sample_products))
    
    # Demo 3: Order Details Template
    print("\n\n3️⃣  ORDER DETAILS TEMPLATE")
    print("-" * 40)
    sample_order = {
        'id': 'ORD12345',
        'customer': 'John Smith',
        'status': 'confirmed',
        'total_amount': 535000,
        'items': [
            {'name': 'iPhone 15 Pro Max', 'quantity': 1, 'price': 450000},
            {'name': 'AirPods Pro 2', 'quantity': 1, 'price': 85000}
        ]
    }
    print(MessageTemplates.order_details(sample_order))
    
    # Demo 4: Tracking Status Template
    print("\n\n4️⃣  TRACKING STATUS TEMPLATE")
    print("-" * 40)
    sample_tracking = {
        'order_id': 'ORD12345',
        'status': 'shipped',
        'estimated_delivery': '2025-09-25'
    }
    print(MessageTemplates.tracking_status(sample_tracking))
    
    # Demo 5: Customer Info Template
    print("\n\n5️⃣  CUSTOMER INFO TEMPLATE")
    print("-" * 40)
    sample_customer = {
        'name': 'John Smith',
        'email': 'john.smith@email.com',
        'phone': '+94771234567',
        'address': '123 Main Street, Colombo 07, Sri Lanka'
    }
    print(MessageTemplates.customer_info(sample_customer))
    
    # Demo 6: Payment Confirmation Template
    print("\n\n6️⃣  PAYMENT CONFIRMATION TEMPLATE")
    print("-" * 40)
    sample_payment = {
        'order_id': 'ORD12345',
        'amount': 535000,
        'method': 'bank_transfer'
    }
    print(MessageTemplates.payment_confirmation(sample_payment))
    
    # Demo 7: Error Messages Template
    print("\n\n7️⃣  ERROR MESSAGES TEMPLATE")
    print("-" * 40)
    print(MessageTemplates.error_message('not_found', 'Product "XYZ Phone" not found in our catalog'))
    print("\n")
    print(MessageTemplates.error_message('out_of_stock', 'iPhone 15 Pro Max is currently out of stock'))
    
    print("\n" + "=" * 60)
    print("✨ All templates are working beautifully! ✨")
    print("=" * 60)

if __name__ == "__main__":
    demo_templates()