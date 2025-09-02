from fastapi import APIRouter, Request, HTTPException, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from typing import Dict, Any
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from utils.logger import get_logger
from services.whatsapp_service import whatsapp_service
# from services.image_analysis_service import deepseek_vision_service
from repositories.tools import get_seller_id_by_whatsapp_number_id
from agent.agent import create_optimized_chatbot
import asyncio
from datetime import datetime
import re
import os

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
        logger.info(response)
        # send images
        img_urls = chatbot.get_img_urls()
        if img_urls:
            for url in img_urls:
                whatsapp_service.send_image_message(phone_number, url, "", whatsapp_number_id)
        # # Remove URLs from response for text message
        # response = remove_urls_from_message(response)

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
        whatsapp_service.send_text_message(phone_number, error_message, whatsapp_number_id)

def process_image_with_deepseek(image_path: str, phone_number: str, whatsapp_number_id: str):
    """Process image with DeepSeek Vision and send analysis"""
    try:
        if not deepseek_vision_service.is_configured():
            logger.warning("DeepSeek Vision service not configured")
            return
        
        logger.info(f"🔍 Analyzing image with DeepSeek Vision: {image_path}")
        
        # Analyze image with business context
        analysis_result = deepseek_vision_service.analyze_image_for_business(
            image_path=image_path,
            business_context="This is a customer inquiry image from WhatsApp business chat.",
            language="en"  # Using English for analysis
        )
        
        if analysis_result["success"]:
            analysis_text = analysis_result["analysis"]
            
            # Send the analysis back to the customer
            response_message = f"🔍 Image Analysis:\n\n{analysis_text}"
            
            result = whatsapp_service.send_text_message(phone_number, response_message, whatsapp_number_id)
            
            if result["success"]:
                logger.info(f"Image analysis sent to {phone_number}")
            else:
                logger.error(f"Failed to send image analysis to {phone_number}")
                
            # Also try to extract text if any
            text_result = deepseek_vision_service.extract_text_from_image(image_path, "en")
            if text_result["success"] and text_result["analysis"].strip():
                text_message = f"Text extracted from image:\n\n{text_result['analysis']}"
                whatsapp_service.send_text_message(phone_number, text_message, whatsapp_number_id)
                
        else:
            logger.error(f"Failed to analyze image: {analysis_result.get('error')}")
            error_message = "Sorry, I couldn't analyze the image. Please try again."
            whatsapp_service.send_text_message(phone_number, error_message, whatsapp_number_id)
            
    except Exception as e:
        logger.error(f"Error processing image with DeepSeek: {str(e)}")
        error_message = "An error occurred while analyzing the image."
        whatsapp_service.send_text_message(phone_number, error_message, whatsapp_number_id)

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

        # Handle image messages with DeepSeek Vision
        if whatsapp_message and whatsapp_message.message_type.value == "image":
            logger.info(f"Received image message from {whatsapp_message.from_number}")
            
            # Download the image
            download_result = whatsapp_service.download_image(whatsapp_message)
            
            if download_result["success"] and download_result["file_path"]:
                logger.info(f"Image downloaded: {download_result['file_path']}")
                

                # Process image with DeepSeek Vision in background
                # background_tasks.add_task(
                #     process_image_with_deepseek,
                #     download_result["file_path"],
                #     whatsapp_message.from_number,
                #     whatsapp_message.to_number
                # )
                # Also send a quick acknowledgment
                ack_message = "Image received, analyzing..."
                whatsapp_service.send_text_message(
                    whatsapp_message.from_number, 
                    ack_message, 
                    whatsapp_message.to_number
                )

                content = f"[Image received: {download_result['file_path']}]"
                background_tasks.add_task(
                process_whatsapp_message_async,
                whatsapp_message.from_number,
                content,
                whatsapp_message.message_id,
                whatsapp_message.to_number
            )
                
        
            else:
                logger.error(f"Failed to download image: {download_result.get('error')}")
                error_message = "Sorry, I couldn't download the image. Please try again."
                whatsapp_service.send_text_message(
                    whatsapp_message.from_number, 
                    error_message, 
                    whatsapp_message.to_number
                )
        
        # Handle text messages normally
        elif whatsapp_message and whatsapp_message.content:
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

@router.post("/analyze-image")
async def analyze_image_endpoint(data: Dict[str, Any]):
    """
    Manual endpoint to analyze images using DeepSeek Vision
    
    Body format:
    {
        "image_path": "/path/to/image.jpg",
        "prompt": "Custom analysis prompt (optional)",
        "language": "en" or "ar",
        "business_context": "Additional business context (optional)"
    }
    """
    try:
        image_path = data.get("image_path")
        prompt = data.get("prompt")
        language = data.get("language", "en")
        business_context = data.get("business_context", "")
        
        if not image_path:
            raise HTTPException(status_code=400, detail="Missing 'image_path' field")
        
        if not deepseek_vision_service.is_configured():
            raise HTTPException(status_code=503, detail="DeepSeek Vision service not configured")
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Perform analysis
        if business_context or not prompt:
            result = deepseek_vision_service.analyze_image_for_business(
                image_path=image_path,
                business_context=business_context,
                language=language
            )
        else:
            result = deepseek_vision_service.analyze_image(
                image_path=image_path,
                prompt=prompt,
                language=language
            )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in image analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-text")
async def extract_text_endpoint(data: Dict[str, Any]):
    """
    Manual endpoint to extract text from images using DeepSeek Vision
    
    Body format:
    {
        "image_path": "/path/to/image.jpg",
        "language": "en" or "ar"
    }
    """
    try:
        image_path = data.get("image_path")
        language = data.get("language", "en")
        
        if not image_path:
            raise HTTPException(status_code=400, detail="Missing 'image_path' field")
        
        if not deepseek_vision_service.is_configured():
            raise HTTPException(status_code=503, detail="DeepSeek Vision service not configured")
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Extract text
        result = deepseek_vision_service.extract_text_from_image(
            image_path=image_path,
            language=language
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in text extraction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vision/status")
async def vision_status():
    """
    Get DeepSeek Vision service status
    """
    return {
        "service": "DeepSeek Vision",
        "configured": deepseek_vision_service.is_configured(),
        "supported_formats": deepseek_vision_service.get_supported_formats(),
        "model": deepseek_vision_service.model,
        "api_available": bool(deepseek_vision_service.api_key)
    }