"""
Main application entry point with global logging configuration
"""
import os
from dotenv import load_dotenv
from utils.logger import GlobalLogger, get_logger

# Load environment variables
load_dotenv()
load_dotenv('.env.logging')  # Load logging configuration

# Initialize global logger
global_logger = GlobalLogger()
logger = get_logger(__name__)

def configure_logging_from_env():
    """Configure logging based on environment variables"""
    
    # Set log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO")
    GlobalLogger.set_log_level(log_level)
    
    # Enable debug mode if specified
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    if debug_mode:
        GlobalLogger.set_debug_mode(True)
        logger.info("Debug mode enabled from environment")
    
    logger.info(f"Logging configured from environment - Level: {log_level}, Debug: {debug_mode}")

def main():
    """Main application entry point"""
    logger.info("Starting Business Chat Assistant")
    
    # Configure logging from environment
    configure_logging_from_env()
    
    # Import and start your FastAPI app here
    # from main import app
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
    logger.info("Application started successfully")

if __name__ == "__main__":
    main()
