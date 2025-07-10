# Performance Configuration for Chatbot

# LLM Settings
LLM_CONFIG = {
    "temperature": 0.1,          # Lower for faster, more deterministic responses
    "max_tokens": 512,           # Limit response length
    "timeout": 30,               # 30 second timeout
    "max_retries": 1,            # Reduce retries
    "stream": False              # Disable streaming for faster single responses
}

# RAG Settings
RAG_CONFIG = {
    "max_examples": 2,           # Reduced from 3-5 to 2
    "cache_size": 100,           # LRU cache size for RAG examples
    "similarity_threshold": 0.7,  # Only include highly relevant examples
    "max_content_length": 200    # Truncate content for speed
}

# Agent Settings
AGENT_CONFIG = {
    "max_iterations": 3,         # Limit agent iterations
    "max_chat_history": 10,      # Limit chat history length
    "enable_verbose": False,     # Disable verbose logging
    "enable_debug": False        # Disable debug logging in production
}

# Cache Settings
CACHE_CONFIG = {
    "intent_cache_size": 50,
    "rag_cache_size": 100,
    "tool_cache_size": 200,
    "cache_ttl": 3600           # 1 hour TTL
}

# Database Settings
DB_CONFIG = {
    "connection_pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30,
    "pool_recycle": 3600
}

# Vector Store Settings
VECTOR_STORE_CONFIG = {
    "lazy_loading": True,        # Load embeddings only when needed
    "batch_size": 100,          # Process in smaller batches
    "max_search_results": 5,    # Limit search results
    "enable_caching": True      # Enable result caching
}

# Logging Settings (for production)
LOGGING_CONFIG = {
    "level": "INFO",            # Set to INFO or WARNING in production
    "disable_debug": True,      # Disable debug logs
    "async_logging": True,      # Use async logging
    "log_rotation": True        # Enable log rotation
}
