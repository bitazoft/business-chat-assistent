import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from services.chat import router as chat_router 
from routes.whatsapp_routes import router as whatsapp_router

# Load environment variables
load_dotenv()

# Initialize custom logger
from utils.logger import GlobalLogger, get_logger
GlobalLogger()  # This sets up the logging configuration
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Business Chat Assistant with WhatsApp Integration")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log application startup
logger.info("🚀 Business Chat Assistant starting up...")
logger.info("✅ CORS middleware configured")

# Include chat routes
app.include_router(chat_router)
logger.info("✅ Chat routes registered")

# Include WhatsApp routes
app.include_router(whatsapp_router)
logger.info("✅ WhatsApp routes registered")

# Health check endpoint
@app.get("/health")
async def health():
    logger.info("Health check requested")
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    logger.info("🎉 Business Chat Assistant with WhatsApp Integration is ready to serve requests!")
    logger.info("📱 WhatsApp webhook endpoint: /whatsapp/webhook")
    logger.info("📊 WhatsApp status endpoint: /whatsapp/status")

@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("👋 Business Chat Assistant is shutting down...")

# uvicorn main:app --reload