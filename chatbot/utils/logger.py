import logging
import os
from datetime import datetime
from typing import Optional

class GlobalLogger:
    """Global logger configuration for the entire application"""
    
    _instance = None
    _configured = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._configured:
            self.setup_logging()
            self._configured = True
    
    def setup_logging(self):
        """Setup global logging configuration"""
        
        # Get log level from environment variable or default to INFO
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        
        # Get debug mode from environment variable
        debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # Set log level based on debug mode
        if debug_mode:
            log_level = "DEBUG"
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                # Console handler
                logging.StreamHandler(),
                # File handler
                logging.FileHandler(
                    os.path.join(log_dir, f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"),
                    encoding='utf-8'
                )
            ]
        )
        
        # Set specific loggers to appropriate levels
        self.configure_module_loggers()
        
        # Log the configuration
        logger = logging.getLogger(__name__)
        logger.info(f"Global logging configured - Level: {log_level}, Debug Mode: {debug_mode}")
    
    def configure_module_loggers(self):
        """Configure specific module loggers"""
        
        # Suppress noisy third-party loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)
        logging.getLogger("langchain_core").setLevel(logging.WARNING)
        logging.getLogger("langchain_openai").setLevel(logging.WARNING)
        logging.getLogger("langchain_deepseek").setLevel(logging.WARNING)
        
        # Set application loggers to appropriate levels
        app_modules = [
            "agent.agent",
            "services.chat", 
            "repositories.tools",
            "db.database",
            "vector_store.vector_store"
        ]
        
        for module in app_modules:
            logger = logging.getLogger(module)
            # Keep application loggers at the global level
            logger.setLevel(logging.getLogger().level)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger for the specified module"""
        return logging.getLogger(name)
    
    @staticmethod
    def set_debug_mode(enabled: bool = True):
        """Enable or disable debug mode globally"""
        level = logging.DEBUG if enabled else logging.INFO
        logging.getLogger().setLevel(level)
        
        # Update all application loggers
        for logger_name in logging.getLogger().manager.loggerDict:
            if any(module in logger_name for module in ["agent", "services", "repositories", "db", "vector_store"]):
                logging.getLogger(logger_name).setLevel(level)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'} globally")
    
    @staticmethod
    def set_log_level(level: str):
        """Set global log level"""
        try:
            log_level = getattr(logging, level.upper())
            logging.getLogger().setLevel(log_level)
            
            # Update all application loggers
            for logger_name in logging.getLogger().manager.loggerDict:
                if any(module in logger_name for module in ["agent", "services", "repositories", "db", "vector_store"]):
                    logging.getLogger(logger_name).setLevel(log_level)
            
            logger = logging.getLogger(__name__)
            logger.info(f"Global log level set to {level.upper()}")
        except AttributeError:
            logger = logging.getLogger(__name__)
            logger.error(f"Invalid log level: {level}")

# Initialize global logger
global_logger = GlobalLogger()

# Convenience function to get logger
def get_logger(name: str) -> logging.Logger:
    """Get a logger for the specified module"""
    return GlobalLogger.get_logger(name)
