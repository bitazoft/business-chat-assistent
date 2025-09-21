# Performance Optimization Guide - Reducing 3s Response Delay

## 🎯 Priority 1: Make Database Logging Asynchronous

### Current Issue:
The `log_query()` function runs synchronously after each response, causing blocking delays.

### Solution 1: Background Task Logging
```python
# In agent.py - Replace synchronous logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create thread pool for background tasks
background_executor = ThreadPoolExecutor(max_workers=2)

def log_query_async(self, query: str, intent: str, response: str, entities: Dict, response_time: int):
    """Log query asynchronously to avoid blocking"""
    def _log_in_background():
        try:
            log_query(
                query=query,
                intent=intent,
                entities=entities,
                response=response,
                seller_id=self.seller_id,
                user_id=self.user_id,
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"Background logging error: {str(e)}")
    
    # Submit to background executor
    background_executor.submit(_log_in_background)

# Replace the synchronous call in process_message():
# OLD: self.log_query(message, intent, response, self.extract_entities(), total_time*1000)
# NEW: self.log_query_async(message, intent, response, self.extract_entities(), total_time*1000)
```

### Solution 2: Use FastAPI Background Tasks
```python
# In routes/whatsapp_routes.py
from fastapi import BackgroundTasks

def log_conversation_background(message: str, response: str, user_id: str, seller_id: str):
    """Background task for logging"""
    # Move logging here
    pass

# In process_whatsapp_message, add background_tasks parameter and use it
```

## 🎯 Priority 2: Optimize WhatsApp API Calls

### Current Issues:
- `mark_message_as_read()` runs synchronously after response
- Image sending happens sequentially

### Solution: Async WhatsApp Operations
```python
# In routes/whatsapp_routes.py - Modify process_whatsapp_message
async def process_whatsapp_message_async(phone_number: str, message_content: str, message_id: str, whatsapp_number_id: str):
    """Async version with optimized API calls"""
    try:
        # Process message first
        chatbot = get_or_create_chatbot(phone_number, seller_id)
        response = chatbot.process_message(message_content)
        
        # Send text response immediately (don't wait)
        text_result = whatsapp_service.send_text_message(phone_number, response, whatsapp_number_id)
        
        # Handle images and read receipts in background
        if text_result["success"]:
            # Background tasks (non-blocking)
            asyncio.create_task(send_images_async(chatbot, phone_number, whatsapp_number_id))
            asyncio.create_task(mark_read_async(message_id, whatsapp_number_id))
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")

async def send_images_async(chatbot, phone_number: str, whatsapp_number_id: str):
    """Send images asynchronously"""
    img_urls = chatbot.get_img_urls()
    if img_urls:
        # Send images concurrently, not sequentially
        tasks = [
            asyncio.create_task(whatsapp_service.send_image_message(phone_number, url, "", whatsapp_number_id))
            for url in img_urls
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

async def mark_read_async(message_id: str, whatsapp_number_id: str):
    """Mark message as read asynchronously"""
    whatsapp_service.mark_message_as_read(message_id, whatsapp_number_id)
```

## 🎯 Priority 3: Reduce Template Processing Overhead

### Solution: Pre-compile Template Checks
```python
# In agent.py - Optimize template detection
import re

# Pre-compile regex patterns (do this once at module level)
TEMPLATE_PATTERNS = [
    re.compile(r'^🛍️'),
    re.compile(r'^🚚'),
    re.compile(r'^📋'),
    # ... other template patterns
]

def is_template_response(text: str) -> bool:
    """Fast template detection using pre-compiled patterns"""
    return any(pattern.match(text) for pattern in TEMPLATE_PATTERNS)

# Replace the template checking loop with:
if tool_results and any(is_template_response(str(result.get("result", ""))) for result in tool_results):
    # Use template response
    pass
```

## 🎯 Priority 4: Optimize Database Operations

### Solution: Connection Pooling & Batch Operations
```python
# In repositories/tools.py - Optimize log_query
def log_query_optimized(query: str, intent: str, entities: Union[str, Dict[str, Any], List], response: str, seller_id: str, user_id: str, response_time: int) -> None:
    """Optimized logging with minimal processing"""
    db = SessionLocal()
    try:
        # Simplified entity processing
        if isinstance(entities, dict):
            entities_json = entities
        else:
            entities_json = {"data": str(entities)}  # Simplified conversion
            
        chat_log = ChatLog(
            user_query=query[:500],  # Truncate long queries
            intent=intent,
            entities=entities_json,
            response=response[:1000],  # Truncate long responses  
            seller_id=int(seller_id),
            customer_id=user_id,
            response_time_ms=response_time
        )
        db.add(chat_log)
        db.commit()
    except Exception as e:
        logger.error(f"Quick log error: {e}")
    finally:
        db.close()
```

## 🎯 Priority 5: Configuration Optimizations

### Immediate Configuration Changes:
```python
# In agent.py - Update LLM config for speed
llm = ChatDeepSeek(
    model=os.getenv("CHAT_MODEL","deepseek-chat"),
    api_key=API_KEY,
    base_url=API_BASE,
    temperature=0.0,      # Changed from 0.1 to 0.0 for fastest responses
    max_tokens=256,       # Reduced from 512 to 256
    timeout=15,           # Reduced from 300 to 15 seconds  
    max_retries=1         # Reduced from 3 to 1
)

# Reduce agent iterations
agent = AgentExecutor(
    agent=agent, 
    tools=self.tools, 
    verbose=False,
    max_iterations=2,     # Reduced from 10 to 2
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False
)
```

### Environment Variables to Set:
```bash
# Add to your .env file
RAG_ENABLED=false                    # Disable RAG for faster responses
LANGUAGE_DETECTION_ENABLED=false    # Disable language detection
ENABLE_DEBUG_LOGGING=false          # Disable verbose logging
MAX_CHAT_HISTORY=5                  # Reduce chat history
```

## 🎯 Priority 6: Quick Wins

### 1. Remove Unnecessary Operations:
```python
# In process_whatsapp_message - Comment out or remove:
# whatsapp_service.mark_message_as_read(message_id, whatsapp_number_id)  # Do this in background

# Simplify image URL extraction:
def get_img_urls_fast(self) -> List[str]:
    """Faster image URL extraction"""
    urls = []
    for result in self.last_tool_results:
        if result.get("tool_name") == "get_product_info":
            urls.extend(re.findall(r'https?://\S+', str(result['result'])))
    return urls[:3]  # Limit to 3 images max
```

### 2. Disable Non-Essential Features:
```python
# In agent.py process_message method:
# Comment out or disable:
# detected_language = detect_language(message)  # Skip language detection
# examples = get_cached_rag_examples(...)       # Skip RAG examples
```

## 📊 Expected Performance Improvements:

| Optimization | Time Saved | Difficulty |
|-------------|------------|------------|
| Async Database Logging | 1-2 seconds | Easy |
| Background WhatsApp API calls | 0.5-1 second | Medium |
| Disable RAG/Language Detection | 0.3-0.5 seconds | Easy |
| Reduce LLM tokens/timeout | 0.2-0.5 seconds | Easy |
| **Total Expected Reduction** | **2-4 seconds** | **Mixed** |

## 🚀 Implementation Priority:

1. **Start with async database logging** (biggest impact)
2. **Disable RAG and language detection** (quick wins)
3. **Reduce LLM parameters** (immediate)
4. **Optimize WhatsApp API calls** (medium effort)
5. **Template processing optimization** (polish)

These optimizations should reduce your 3-second delay to under 1 second in most cases.