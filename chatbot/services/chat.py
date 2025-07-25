from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from agent.agent import create_optimized_chatbot
from agent.agent import log_query, check_user_exists, create_tmp_user_id
from utils.logger import get_logger
import time

# Get logger for this module
logger = get_logger(__name__)

router = APIRouter()

# Pydantic model for chat request
class ChatRequest(BaseModel):
    message: str
    session_id: str
    seller_id: str
    user_id: str
    chat_history: List[Dict[str, str]] = []  # Format: [{"role": "user", "content": "message"}, ...]

# Chat endpoint
@router.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"[Chat API] Received chat request - seller_id: {request.seller_id}, user_id: {request.user_id}, session_id: {request.session_id}")
    logger.debug(f"[Chat API] Message: {request.message}")
    logger.debug(f"[Chat API] Chat history length: {len(request.chat_history)}")
    
    try:
        # Validate seller_id
        logger.info("[Chat API] Validating seller_id")
        if not request.seller_id.isdigit():
            logger.error(f"[Chat API] Invalid seller_id: {request.seller_id} - must be numeric")
            raise HTTPException(status_code=400, detail="Invalid seller_id: must be a numeric ID")
        logger.debug(f"[Chat API] seller_id validation passed: {request.seller_id}")

        # Validate user_id
        logger.info("[Chat API] Validating user_id")
        if not request.user_id:
            logger.error("[Chat API] Invalid user_id: cannot be empty")
            raise HTTPException(status_code=400, detail="Invalid user_id: cannot be empty")
        logger.debug(f"[Chat API] user_id validation passed: {request.user_id}")
        
        # Check if user exists
        logger.info("[Chat API] Checking if user exists")
        user_id = request.user_id
        
        # user_exists = check_user_exists(user_id)
        # logger.debug(f"[Chat API] User exists check result: {user_exists}")
        
        # if not user_exists:
        #     logger.info("[Chat API] User does not exist, creating temporary user ID")
        #     user_id = create_tmp_user_id()
        #     logger.info(f"[Chat API] Created temporary user_id: {user_id}")
        # else:
        #     logger.info(f"[Chat API] Using existing user_id: {user_id}")
            
        
        # Format chat history for LangChain
        logger.info("[Chat API] Formatting chat history")
        formatted_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in request.chat_history
        ]
        logger.debug(f"[Chat API] Formatted history: {formatted_history}")

        # Create optimized chatbot instance
        logger.info("[Chat API] Creating optimized chatbot")
        start_time = time.time()
        
        chatbot = create_optimized_chatbot(request.seller_id, user_id)
        logger.info(f"[Chat API] Chatbot created in {time.time() - start_time:.2f}s")
        
        # Process the message
        logger.info("[Chat API] Processing user input through optimized chatbot")
        start_time = time.time()
        
        response = chatbot.process_message(request.message, formatted_history)
        
        processing_time = time.time() - start_time
        logger.info(f"[Chat API] Message processed in {processing_time:.2f}s")
        logger.debug(f"[Chat API] Response: {response}")

        return {"response": response, "processing_time": f"{processing_time:.2f}s"}
        
    except ValueError as ve:
        logger.error(f"[Chat API] ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except HTTPException as he:
        logger.error(f"[Chat API] HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"[Chat API] Unexpected error: {str(e)}")
        logger.exception("[Chat API] Exception details:")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")