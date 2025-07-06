from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from agent.agent import create_agent_executor, log_query, check_user_exists, create_tmp_user_id
from agent.multi_agent import create_multi_agent_system

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
    try:
        # Validate seller_id
        if not request.seller_id.isdigit():
            raise HTTPException(status_code=400, detail="Invalid seller_id: must be a numeric ID")

        # Validate user_id
        if not request.user_id:
            raise HTTPException(status_code=400, detail="Invalid user_id: cannot be empty")
        
        # Check if user exists
        user_id = request.user_id
        if not check_user_exists(user_id):
            user_id = create_tmp_user_id() 
            
        
        # Format chat history for LangChain
        formatted_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in request.chat_history
        ]

        # Create agent executor with seller_id and user_id bound to tools
        # agent_executor = create_agent_executor(request.seller_id,user_id)
        system = create_multi_agent_system(seller_id="seller_123", user_id="user_123")
        agent_executor = system["executor"]
        # chat_history = system["chat_history"]
        # response = agent_executor({"input": "Book a meeting for tomorrow at 10 AM with Alice"}, chat_history)
        print(f"Agent executor created for seller_id: {request.seller_id}, user_id: {user_id}")
        # Pass seller_id to agent for seller-specific queries
        response = agent_executor.invoke({
            "input": request.message,
            "chat_history": formatted_history
        })

        return {"response": response["output"]}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")