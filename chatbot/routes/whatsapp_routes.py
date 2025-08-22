from fastapi import APIRouter, Request, HTTPException, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from typing import Dict, Any
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from utils.logger import get_logger
from services.whatsapp_service import whatsapp_service
from repositories.tools import get_seller_id_by_whatsapp_number_id
from agent.agent import create_optimized_chatbot
import asyncio
from datetime import datetime

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/whatsapp", tags=["WhatsApp"])

# Store active chatbot sessions
active_sessions: Dict[str, Any] = {}
# Thread pool configuration
MAX_THREADS = 10  # Configurable maximum number of concurrent threads
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
# Lock for thread-safe session management
sessions_lock = threading.Lock()

def get_or_create_chatbot(phone_number: str, seller_id: str = "default_seller"):
    """Get existing chatbot session or create new one"""
    session_key = f"{seller_id}_{phone_number}"
    
    with sessions_lock:
        if session_key not in active_sessions:
            logger.info(f"Creating new chatbot session for {phone_number}")
            active_sessions[session_key] = {
                "chatbot": create_optimized_chatbot(seller_id=seller_id, user_id=phone_number),
                "last_activity": datetime.now(),
                "message_count": 0
            }
        else:
            # Update last activity
            active_sessions[session_key]["last_activity"] = datetime.now()
            active_sessions[session_key]["message_count"] += 1
    
    return active_sessions[session_key]["chatbot"]

def process_whatsapp_message(phone_number: str, message_content: str, message_id: str, whatsapp_number_id: str = "default_seller"):
    """Process WhatsApp message in a separate thread"""
    try:
        logger.info(f"🤖 Processing message from {phone_number}: {message_content[:50]}...")
        
        seller_id = get_seller_id_by_whatsapp_number_id(whatsapp_number_id)
        # Get or create chatbot for this user
        chatbot = get_or_create_chatbot(phone_number, seller_id)
        
        # Process the message through the chatbot
        response = chatbot.process_message(message_content)
        
        # Send response back to WhatsApp
        result = whatsapp_service.send_text_message(phone_number, response, whatsapp_number_id)
        
        if result["success"]:
            logger.info(f"✅ Response sent to {phone_number}: {response[:50]}...")
            # Mark original message as read
            whatsapp_service.mark_message_as_read(message_id, whatsapp_number_id)
        else:
            logger.error(f"❌ Failed to send response to {phone_number}: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"❌ Error processing WhatsApp message: {str(e)}")
        # Send error message to user
        error_message = "I'm experiencing technical difficulties. Please try again in a moment."
        whatsapp_service.send_text_message(phone_number, error_message)

async def process_whatsapp_message_async(phone_number: str, message_content: str, message_id: str, whatsapp_number_id: str = "default_seller"):
    """Submit message processing to thread pool"""
    # Run the synchronous processing in thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        thread_pool,
        process_whatsapp_message,
        phone_number,
        message_content,
        message_id,
        whatsapp_number_id
    )

@router.get("/")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"), 
    hub_challenge: str = Query(alias="hub.challenge")
):
    """
    Webhook verification endpoint for WhatsApp
    This endpoint is called by WhatsApp to verify your webhook URL
    """
    logger.info(f"🔐 Webhook verification attempt - Mode: {hub_mode}, Token: {hub_verify_token[:10]}...")
    
    if hub_mode == "subscribe":
        # Verify the token
        challenge = whatsapp_service.verify_webhook(hub_verify_token, hub_challenge)
        if challenge:
            logger.info("✅ Webhook verification successful")
            return PlainTextResponse(challenge)
        else:
            logger.warning("❌ Webhook verification failed")
            raise HTTPException(status_code=403, detail="Verification failed")
    
    raise HTTPException(status_code=400, detail="Invalid request")

@router.post("/")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming WhatsApp messages
    This endpoint receives all WhatsApp events (messages, status updates, etc.)
    """
    try:
        # Get the raw request body
        body = await request.body()
        webhook_data = json.loads(body.decode('utf-8'))
        
        logger.info(f"📨 Received webhook data: {json.dumps(webhook_data, indent=2)}...")
        
        # Parse the message
        whatsapp_message = whatsapp_service.parse_webhook_message(webhook_data)
        
        if whatsapp_message and whatsapp_message.content:
            # Process message in background to respond quickly
            background_tasks.add_task(
                process_whatsapp_message_async,
                whatsapp_message.from_number,
                whatsapp_message.content,
                whatsapp_message.message_id,
                whatsapp_message.to_number
            )
            
            logger.info(f"✅ Message queued for processing from {whatsapp_message.from_number} , to {whatsapp_message.to_number}")
        else:
            logger.info("📝 Webhook received but no message to process (might be status update)")
        
        # Always return 200 to acknowledge receipt
        return {"status": "received"}
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in webhook: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"❌ Error handling webhook: {str(e)}")
        # Still return 200 to prevent WhatsApp from retrying
        return {"status": "error", "message": str(e)}

@router.post("/send-message")
async def send_message(data: Dict[str, Any]):
    """
    Manual endpoint to send messages (for testing or admin use)
    
    Body format:
    {
        "to": "1234567890",
        "message": "Hello from the chatbot!",
        "type": "text"
    }
    """
    try:
        to_number = data.get("to")
        message = data.get("message")
        message_type = data.get("type", "text")
        
        if not to_number or not message:
            raise HTTPException(status_code=400, detail="Missing 'to' or 'message' field")
        
        if message_type == "text":
            result = whatsapp_service.send_text_message(to_number, message)
        elif message_type == "image":
            image_url = data.get("image_url")
            caption = data.get("caption", "")
            if not image_url:
                raise HTTPException(status_code=400, detail="Missing 'image_url' for image message")
            result = whatsapp_service.send_image_message(to_number, image_url, caption)
        else:
            raise HTTPException(status_code=400, detail="Unsupported message type")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in manual send message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """
    Get service status and active sessions
    """
    with sessions_lock:
        return {
            "status": "active",
            "active_sessions": len(active_sessions),
            "active_threads": thread_pool._work_queue.qsize(),
            "max_threads": MAX_THREADS,
            "sessions": {
                session_key: {
                    "last_activity": session_data["last_activity"].isoformat(),
                    "message_count": session_data["message_count"]
                }
                for session_key, session_data in active_sessions.items()
            }
        }

@router.delete("/sessions/{phone_number}")
async def clear_session(phone_number: str, seller_id: str = "default_seller"):
    """
    Clear a specific user session
    """
    session_key = f"{seller_id}_{phone_number}"
    with sessions_lock:
        if session_key in active_sessions:
            del active_sessions[session_key]
            logger.info(f"🗑️ Cleared session for {phone_number}")
            return {"status": "session_cleared", "phone_number": phone_number}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

@router.delete("/sessions")
async def clear_all_sessions():
    """
    Clear all active sessions
    """
    global active_sessions
    with sessions_lock:
        session_count = len(active_sessions)
        active_sessions.clear()
        logger.info(f"🗑️ Cleared {session_count} sessions")
        return {"status": "all_sessions_cleared", "cleared_count": session_count}

@router.get("/profile/{phone_number}")
async def get_profile(phone_number: str):
    """
    Get WhatsApp profile information for a phone number
    """
    try:
        result = whatsapp_service.get_profile_info(phone_number)
        return result
    except Exception as e:
        logger.error(f"❌ Error getting profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))