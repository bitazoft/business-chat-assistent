import os
from fastapi import FastAPI
from dotenv import load_dotenv
from services.chat import router as chat_router 

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Business Chat Assistant")

# Include chat routes
app.include_router(chat_router)

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

# uvicorn main:app --reload