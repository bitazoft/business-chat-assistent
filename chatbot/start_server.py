#!/usr/bin/env python3
"""
Optimized startup script for the chatbot application
This script applies performance optimizations before starting the server
"""

import os
import sys
import uvicorn
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize custom logger FIRST
from utils.logger import GlobalLogger
GlobalLogger()  # This sets up the logging configuration

def setup_performance_optimizations():
    """Apply performance optimizations"""
    
    # 1. Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer warnings
    os.environ["HF_HOME"] = "cache/huggingface"  # Cache HuggingFace models (includes transformers)
    
    # 2. Configure logging for production (but don't suppress app logs)
    if os.getenv("ENVIRONMENT") == "production":
        # Only suppress uvicorn access logs in production, keep other logs visible
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # 3. Set optimized Python flags
    sys.dont_write_bytecode = False  # Enable bytecode caching
    
    print("✅ Performance optimizations applied")

def create_cache_directories():
    """Create necessary cache directories"""
    cache_dirs = [
        "cache",
        "cache/transformers", 
        "cache/huggingface",
        "cache/vector_store",
        "cache/embeddings"
    ]
    
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
    
    print("✅ Cache directories created")

def preload_models():
    """Preload models and vector stores for faster first response"""
    try:
        print("🔄 Preloading models and vector stores...")
        
        # Import and initialize the fast vector store
        from vector_store.vector_store import fast_vector_store
        fast_vector_store._lazy_load()
        
        # Preload the LLM by making a test call
        from chatbot.agent.agent import llm
        test_response = llm.invoke("Hello")
        
        print("✅ Models preloaded successfully")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not preload models: {str(e)}")
        print("   The application will still work, but first response may be slower")

def main():
    """Main startup function"""
    print("🚀 Starting Optimized Chatbot Server")
    print("=" * 40)
    
    # Apply optimizations
    setup_performance_optimizations()
    create_cache_directories()
    
    # Preload models (optional, can be skipped for faster startup)
    if os.getenv("PRELOAD_MODELS", "true").lower() == "true":
        preload_models()
    else:
        print("⏭️  Skipping model preloading (set PRELOAD_MODELS=true to enable)")
    
    # Configure uvicorn settings
    config = {
        "app": "main:app",
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", 8080)),
        "reload": os.getenv("ENVIRONMENT", "development") != "production",
        "workers": 1,  # Single worker for development, increase for production
        "access_log": True,  # Always show access logs
        "log_level": "info"  # Always use info level for uvicorn
    }
    
    print(f"🌐 Server starting on {config['host']}:{config['port']}")
    print("📊 Performance monitoring available at /health")
    print("🔧 Use performance_test.py to measure response times")
    print("=" * 40)
    
    # Start the server
    uvicorn.run(**config)

if __name__ == "__main__":
    main()
