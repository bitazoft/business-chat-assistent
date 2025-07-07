"""
Runtime logging control utilities
"""
from utils.logger import GlobalLogger, get_logger
from fastapi import APIRouter

logger = get_logger(__name__)
router = APIRouter()

@router.post("/admin/logging/debug/enable")
async def enable_debug_logging():
    """Enable debug logging globally"""
    GlobalLogger.set_debug_mode(True)
    logger.info("Debug logging enabled via admin endpoint")
    return {"status": "success", "message": "Debug logging enabled"}

@router.post("/admin/logging/debug/disable")
async def disable_debug_logging():
    """Disable debug logging globally"""
    GlobalLogger.set_debug_mode(False)
    logger.info("Debug logging disabled via admin endpoint")
    return {"status": "success", "message": "Debug logging disabled"}

@router.post("/admin/logging/level/{level}")
async def set_log_level(level: str):
    """Set global log level"""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    if level.upper() not in valid_levels:
        return {"status": "error", "message": f"Invalid log level. Valid levels: {valid_levels}"}
    
    GlobalLogger.set_log_level(level)
    logger.info(f"Log level set to {level.upper()} via admin endpoint")
    return {"status": "success", "message": f"Log level set to {level.upper()}"}

@router.get("/admin/logging/status")
async def get_logging_status():
    """Get current logging configuration"""
    import logging
    
    root_logger = logging.getLogger()
    current_level = logging.getLevelName(root_logger.level)
    
    return {
        "status": "success",
        "current_level": current_level,
        "is_debug": current_level == "DEBUG",
        "handlers": len(root_logger.handlers)
    }
