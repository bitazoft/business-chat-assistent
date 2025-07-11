import os
from fastapi import FastAPI
from dotenv import load_dotenv
from services.chat import router as chat_router 

# Load environment variables
load_dotenv()

# Initialize custom logger
from utils.logger import GlobalLogger, get_logger
GlobalLogger()  # This sets up the logging configuration
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Business Chat Assistant")

# Log application startup
logger.info("🚀 Business Chat Assistant starting up...")

# Include chat routes
app.include_router(chat_router)
logger.info("✅ Chat routes registered")

# Health check endpoint
@app.get("/health")
async def health():
    logger.info("Health check requested")
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    logger.info("🎉 Business Chat Assistant is ready to serve requests!")

@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("👋 Business Chat Assistant is shutting down...")

# uvicorn main:app --reload