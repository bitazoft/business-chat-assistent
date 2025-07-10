#!/usr/bin/env python3
"""
Test script to verify logging configuration is working
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize logger
from utils.logger import GlobalLogger, get_logger
GlobalLogger()
logger = get_logger(__name__)

def test_logging():
    """Test all logging levels"""
    print("🧪 Testing logging configuration...")
    print("=" * 50)
    
    # Test different log levels
    logger.debug("🐛 This is a DEBUG message")
    logger.info("ℹ️  This is an INFO message")
    logger.warning("⚠️  This is a WARNING message")
    logger.error("❌ This is an ERROR message")
    
    # Test from different modules
    from agent.agent import logger as agent_logger
    agent_logger.info("📤 Log message from agent module")
    
    from services.chat import logger as chat_logger  
    chat_logger.info("💬 Log message from chat service")
    
    print("=" * 50)
    print("✅ Logging test completed!")
    print("📁 Check the logs/ directory for log files")
    print("📋 You should see logs both in console and in log files")

if __name__ == "__main__":
    test_logging()
