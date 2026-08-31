# Beautiful templates with emojis for displaying information
class MessageTemplates:
    """Beautiful message templates with emojis for product and order displays"""
    
    @staticmethod
    def product_details(product_data):
        """Template for displaying individual product details"""
        try:
            # Parse product data from string format
            if isinstance(product_data, str):
                # Extract information from string format like "Product ID: 1, Product: Name, Description: ..., Price: Rs.100, Stock: 50"
                parts = {}
                for part in product_data.split(', '):
                    if ': ' in part:
                        key, value = part.split(': ', 1)
                        parts[key.lower().replace(' ', '_')] = value
                
                product_id = parts.get('product_id', 'N/A')
                name = parts.get('product', 'Unknown Product')
                description = parts.get('description', 'No description available')
                price = parts.get('price', 'N/A')
                stock = parts.get('stock', 'N/A')
            else:
                # Handle dictionary format.
                # get_product_info returns 'product_id' and 'product'; this only
                # looked for 'id' and 'name', so every product was shown to the
                # customer as "Product ID: N/A / Name: Unknown Product". Both
                # spellings are accepted now.
                product_id = product_data.get('product_id', product_data.get('id', 'N/A'))
                name = product_data.get('product') or product_data.get('name') or 'Unknown Product'
                description = product_data.get('description', 'No description available')
                price = f"Rs.{product_data.get('price', 0)}"
                stock = product_data.get('stock', 'N/A')
            
            template = f"""
🛍️ *PRODUCT DETAILS*
━━━━━━━━━━━━━━━━━━━━
🏷️ *Product ID:* {product_id}
📦 *Name:* {name}
📝 *Description:* {description}
💰 *Price:* {price}
📊 *Stock:* {stock} units
━━━━━━━━━━━━━━━━━━━━
✨ *Ready to order? Just let me know!* ✨
            """.strip()
            return template
        except Exception as e:
            return f"🛍️ *PRODUCT DETAILS*\n━━━━━━━━━━━━━━━━━━━━\n❌ Error displaying product: {str(e)}"
    
    @staticmethod
    def product_list(products_data):
        """Template for displaying multiple products"""
        try:
            if isinstance(products_data, str):
                # Handle string format - split by newlines or commas
                products = products_data.strip().split('\n') if '\n' in products_data else products_data.split(', ')
            else:
                products = products_data if isinstance(products_data, list) else [products_data]
            
            template = """
🛒 *PRODUCT CATALOG*
━━━━━━━━━━━━━━━━━━━━
"""
            
            for i, product in enumerate(products[:10], 1):  # Limit to 10 products
                if isinstance(product, str):
                    # Handle string format
                    if product.strip():
                        template += f"📦 *{i}. {product.strip()}*\n"
                else:
                    # Handle dictionary format
                    name = product.get('name', 'Unknown Product')
                    price = product.get('price', 'N/A')
                    stock = product.get('stock', 'N/A')
                    template += f"📦 *{i}. {name} - Rs.{price} (Stock: {stock})*\n"
            
            template += """━━━━━━━━━━━━━━━━━━━━
🔍 *Need details? Ask about any specific product!*
💬 *Ready to order? Just tell me what you want!*"""
            
            return template.strip()
        except Exception as e:
            return f"🛒 *PRODUCT CATALOG*\n━━━━━━━━━━━━━━━━━━━━\n❌ Error displaying products: {str(e)}"
    
    @staticmethod
    def order_details(order_data):
        """Template for displaying order details"""
        try:
            if isinstance(order_data, str):
                # Parse order data from string
                template = f"""
📋 *ORDER DETAILS*
━━━━━━━━━━━━━━━━━━━━
{order_data}
━━━━━━━━━━━━━━━━━━━━
✅ *Your order has been placed successfully!*
📞 *We'll contact you for confirmation*
                """.strip()
            else:
                # Handle dictionary format
                order_id = order_data.get('order_id', 'N/A')
                total = order_data.get('total_amount', 0)
                status = order_data.get('status', 'pending')
                items = order_data.get('items', [])
                
                # Status emoji mapping
                status_emoji = {
                    'pending': '⏳',
                    'confirmed': '✅',
                    'processing': '🔄',
                    'shipped': '🚚',
                    'delivered': '📦',
                    'cancelled': '❌'
                }
                
                template = f"""
📋 *ORDER DETAILS*
━━━━━━━━━━━━━━━━━━━━
🆔 *Order ID:* #{order_id}
{status_emoji.get(status.lower(), '📋')} *Status:* {status.title()}
💰 *Total Amount:* Rs.{total}

📦 *Items:*"""
                
                if items:
                    for i, item in enumerate(items, 1):
                        item_name = item.get('name', item.get('product', 'Unknown'))
                        quantity = item.get('quantity', 1)
                        price = item.get('price', 0)
                        template += f"\n   {i}. {item_name} × {quantity} = Rs.{price * quantity}"
                
                template += """
━━━━━━━━━━━━━━━━━━━━
✨ *Thank you for your order!* ✨"""
                
            return template.strip()
        except Exception as e:
            return f"📋 *ORDER DETAILS*\n━━━━━━━━━━━━━━━━━━━━\n❌ Error displaying order: {str(e)}"
    
    @staticmethod
    def tracking_status(tracking_data):
        """Template for order tracking status"""
        try:
            if isinstance(tracking_data, str):
                # Handle string format
                template = f"""
🚚 *ORDER TRACKING*
━━━━━━━━━━━━━━━━━━━━
{tracking_data}
━━━━━━━━━━━━━━━━━━━━
📱 *Stay tuned for updates!*
                """.strip()
            else:
                order_id = tracking_data.get('order_id', 'N/A')
                status = tracking_data.get('status', 'pending').lower()
                estimated_delivery = tracking_data.get('estimated_delivery', 'TBD')
                
                # Tracking progress with emojis
                tracking_steps = {
                    'pending': '⏳ Order Placed',
                    'confirmed': '✅ Order Confirmed',
                    'processing': '🔄 Processing',
                    'packed': '📦 Packed',
                    'shipped': '🚚 Shipped',
                    'out_for_delivery': '🚛 Out for Delivery',
                    'delivered': '✅ Delivered',
                    'cancelled': '❌ Cancelled'
                }
                
                # Create progress indicator
                current_step = status
                progress_indicator = ""
                
                steps_order = ['pending', 'confirmed', 'processing', 'packed', 'shipped', 'out_for_delivery', 'delivered']
                current_index = steps_order.index(current_step) if current_step in steps_order else 0
                
                for i, step in enumerate(steps_order):
                    if i <= current_index:
                        progress_indicator += f"✅ {tracking_steps[step]}\n"
                    else:
                        progress_indicator += f"⭕ {tracking_steps[step]}\n"
                
                template = f"""
🚚 *ORDER TRACKING*
━━━━━━━━━━━━━━━━━━━━
🆔 *Order ID:* #{order_id}
📍 *Current Status:* {status.replace('_', ' ').title()}
📅 *Estimated Delivery:* {estimated_delivery}

📊 *Progress:*
{progress_indicator}
━━━━━━━━━━━━━━━━━━━━
📱 *We'll notify you of any updates!*
                """.strip()
                
            return template
        except Exception as e:
            return f"🚚 *ORDER TRACKING*\n━━━━━━━━━━━━━━━━━━━━\n❌ Error displaying tracking: {str(e)}"
    
    @staticmethod
    def customer_info(customer_data):
        """Template for displaying customer information"""
        try:
            if isinstance(customer_data, str):
                template = f"""
👤 *CUSTOMER INFORMATION*
━━━━━━━━━━━━━━━━━━━━
{customer_data}
━━━━━━━━━━━━━━━━━━━━
✏️ *Need to update? Just let me know!*
                """.strip()
            else:
                name = customer_data.get('name', 'N/A')
                email = customer_data.get('email', 'N/A')
                phone = customer_data.get('phone', customer_data.get('number1', 'N/A'))
                address = customer_data.get('address', 'N/A')
                
                template = f"""
👤 *CUSTOMER INFORMATION*
━━━━━━━━━━━━━━━━━━━━
📛 *Name:* {name}
📧 *Email:* {email}
📞 *Phone:* {phone}
🏠 *Address:* {address}
━━━━━━━━━━━━━━━━━━━━
✏️ *Need to update? Just let me know!*
                """.strip()
                
            return template
        except Exception as e:
            return f"👤 *CUSTOMER INFORMATION*\n━━━━━━━━━━━━━━━━━━━━\n❌ Error displaying customer info: {str(e)}"
    
    @staticmethod
    def payment_confirmation(payment_data):
        """Template for payment confirmation"""
        try:
            payment_method = payment_data.get('method', 'Unknown')
            amount = payment_data.get('amount', 0)
            order_id = payment_data.get('order_id', 'N/A')
            
            method_emoji = {
                'cod': '💵',
                'cash_on_delivery': '💵',
                'bank_transfer': '🏦',
                'card': '💳',
                'online': '💻'
            }
            
            emoji = method_emoji.get(payment_method.lower(), '💰')
            
            template = f"""
{emoji} *PAYMENT CONFIRMATION*
━━━━━━━━━━━━━━━━━━━━
🆔 *Order ID:* #{order_id}
💰 *Amount:* Rs.{amount}
💳 *Payment Method:* {payment_method.replace('_', ' ').title()}
✅ *Status:* Confirmed

📋 *Next Steps:*
• Your order is being processed
• You'll receive updates via SMS/WhatsApp
• Estimated delivery: 2-3 business days

━━━━━━━━━━━━━━━━━━━━
🙏 *Thank you for your business!*
            """.strip()
            
            return template
        except Exception as e:
            return f"💰 *PAYMENT CONFIRMATION*\n━━━━━━━━━━━━━━━━━━━━\n❌ Error displaying payment info: {str(e)}"
    
    @staticmethod
    def error_message(error_type, details=""):
        """Template for error messages"""
        error_emojis = {
            'not_found': '🔍',
            'out_of_stock': '📦',
            'invalid_input': '⚠️',
            'system_error': '🔧',
            'payment_error': '💳'
        }
        
        emoji = error_emojis.get(error_type, '❌')
        
        error_messages = {
            'not_found': 'Item not found',
            'out_of_stock': 'Out of stock',
            'invalid_input': 'Invalid input provided',
            'system_error': 'System temporarily unavailable',
            'payment_error': 'Payment processing error'
        }
        
        message = error_messages.get(error_type, 'An error occurred')
        
        template = f"""
{emoji} *ERROR*
━━━━━━━━━━━━━━━━━━━━
{message}
{details}
━━━━━━━━━━━━━━━━━━━━
💬 *Please try again or contact support*
        """.strip()
        
        return template

# Template integration helper functions
def format_product_info_response(product_data):
    """Format product info using beautiful template"""
    return MessageTemplates.product_details(product_data)

def format_product_list_response(products_data):
    """Format product list using beautiful template"""  
    return MessageTemplates.product_list(products_data)

def format_order_details_response(order_data):
    """Format order details using beautiful template"""
    return MessageTemplates.order_details(order_data)

def format_tracking_response(tracking_data):
    """Format tracking info using beautiful template"""
    return MessageTemplates.tracking_status(tracking_data)

def format_customer_info_response(customer_data):
    """Format customer info using beautiful template"""
    return MessageTemplates.customer_info(customer_data)

def format_payment_confirmation_response(payment_data):
    """Format payment confirmation using beautiful template"""
    return MessageTemplates.payment_confirmation(payment_data)

def format_error_response(error_type, details=""):
    """Format error message using beautiful template"""
    return MessageTemplates.error_message(error_type, details)
