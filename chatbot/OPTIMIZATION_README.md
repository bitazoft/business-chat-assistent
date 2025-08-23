# Chatbot Model Loading Optimization

## Problem
The SentenceTransformer model `all-MiniLM-L6-v2` was taking ~4 seconds to load on every request, causing slow response times.

## Solutions Implemented

### 1. Model Singleton Cache
- **File**: `vector_store/vector_store.py`
- **Change**: Created `ModelCache` singleton class to cache the SentenceTransformer model in memory
- **Benefit**: Model is loaded only once and reused across requests

### 2. Separated Model and Data Loading
- **Change**: Separated embedding model loading from vector data loading
- **Benefit**: Model loads independently and can be cached separately

### 3. Preloading System
- **Files**: `start_server.py`, `warmup_models.py`
- **Change**: Added preloading during server startup
- **Benefit**: First request is as fast as subsequent requests

### 4. Enhanced Caching
- **File**: `cache/model_cache.py`
- **Change**: Added persistent model caching to disk (optional)
- **Benefit**: Models can persist across application restarts

### 5. Performance Configuration
- **File**: `config/performance.py`
- **Change**: Added model cache configuration options
- **Benefit**: Easy to tune performance settings

## Usage

### Quick Start (Recommended)
The optimizations are automatically applied when you start the server:

```bash
python start_server.py
```

### Manual Model Warmup
If you want to warm up models before starting the server:

```bash
# Warm up all models
python warmup_models.py

# Warm up only vector store
python warmup_models.py --vector-store-only

# Test performance
python warmup_models.py --test-only

# Clear caches and warm up
python warmup_models.py --clear-cache
```

### Environment Variables
Set these for optimal performance:

```bash
export PRELOAD_MODELS=true          # Enable model preloading (default: true)
export ENVIRONMENT=production       # Reduce logging noise
export HF_HOME=cache/huggingface   # Cache directory for models
```

## Expected Performance Improvements

### Before Optimization
- **First Request**: ~4-6 seconds (model loading + search)
- **Subsequent Requests**: ~4-6 seconds (model reloaded each time)

### After Optimization
- **Server Startup**: ~4-6 seconds (one-time model loading)
- **First Request**: ~0.1-0.3 seconds (search only)
- **Subsequent Requests**: ~0.1-0.3 seconds (cached results)

## Configuration Options

### Disable Preloading (faster startup, slower first request)
```bash
export PRELOAD_MODELS=false
python start_server.py
```

### Enable Persistent Caching
```python
# In config/performance.py
MODEL_CACHE_CONFIG = {
    "enabled": True,
    "cache_dir": "cache/models",
    "max_age_hours": 24
}
```

## Monitoring

Check the logs for model loading times:
```
[ModelCache] Loading embedding model all-MiniLM-L6-v2...
[ModelCache] Embedding model all-MiniLM-L6-v2 loaded in 3.94s
[FastVectorStore] Preloading completed in 4.12s
```

## Troubleshooting

### Model Still Loading Slowly
1. Check if preloading is enabled: `export PRELOAD_MODELS=true`
2. Verify cache directories exist and are writable
3. Check for memory constraints (model requires ~500MB RAM)

### Memory Usage
The optimization uses more memory but provides faster responses:
- **Additional RAM**: ~500MB for cached model
- **Disk Space**: ~200MB for model cache (if enabled)

### Clearing Caches
If you encounter issues:
```bash
# Clear all caches
python warmup_models.py --clear-cache

# Or manually remove cache directories
rm -rf cache/models cache/transformers cache/huggingface
```

## Advanced Configuration

### Custom Model Caching
```python
from vector_store.vector_store import ModelCache

# Get cached model directly
model_cache = ModelCache()
model = model_cache.get_model("your-model-name")
```

### Custom Cache Location
```python
# Set custom cache directory
os.environ["HF_HOME"] = "/path/to/your/cache"
os.environ["TRANSFORMERS_CACHE"] = "/path/to/your/cache"
```

## Files Modified

1. `vector_store/vector_store.py` - Main optimization implementation
2. `start_server.py` - Added preloading
3. `config/performance.py` - Performance configuration
4. `cache/model_cache.py` - Persistent caching utility
5. `warmup_models.py` - Manual warmup script

The optimization maintains backward compatibility and can be easily disabled if needed.
