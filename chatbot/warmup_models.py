#!/usr/bin/env python3
"""
Model Warmup Script for Chatbot Application
Run this script to preload models and reduce first request latency
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_cache_directories():
    """Create necessary cache directories"""
    cache_dirs = [
        "cache",
        "cache/models",
        "cache/transformers", 
        "cache/huggingface",
        "cache/vector_store",
        "cache/embeddings"
    ]
    
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
    
    print("✅ Cache directories created")

def warm_up_vector_store():
    """Warm up the vector store and embedding model"""
    print("🔄 Warming up vector store...")
    start_time = time.time()
    
    try:
        from vector_store.vector_store import preload_vector_store
        preload_vector_store()
        
        load_time = time.time() - start_time
        print(f"✅ Vector store warmed up in {load_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Error warming up vector store: {str(e)}")
        return False

def warm_up_llm():
    """Warm up the LLM"""
    print("🔄 Warming up LLM...")
    start_time = time.time()
    
    try:
        from agent.agent import llm
        # Make a simple test call to warm up the model
        response = llm.invoke("Hello")
        
        load_time = time.time() - start_time
        print(f"✅ LLM warmed up in {load_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Error warming up LLM: {str(e)}")
        return False

def test_performance():
    """Test performance after warmup"""
    print("🔄 Testing performance...")
    
    try:
        from vector_store.vector_store import fast_vector_store
        
        # Test vector search
        test_queries = [
            "What are your products?",
            "How can I track my order?",
            "What is your return policy?"
        ]
        
        total_time = 0
        for query in test_queries:
            start_time = time.time()
            results = fast_vector_store.similarity_search(query, k=3)
            search_time = time.time() - start_time
            total_time += search_time
            print(f"  Query: '{query}' - {search_time:.3f}s - {len(results)} results")
        
        avg_time = total_time / len(test_queries)
        print(f"✅ Average search time: {avg_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing performance: {str(e)}")
        return False

def optimize_environment():
    """Set environment variables for better performance"""
    optimizations = {
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": "cache/huggingface",
        "TRANSFORMERS_CACHE": "cache/transformers",
        "SENTENCE_TRANSFORMERS_HOME": "cache/transformers"
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
    
    print("✅ Environment optimized")

def clear_caches():
    """Clear all model caches"""
    print("🔄 Clearing model caches...")
    
    try:
        from cache.model_cache import persistent_cache
        persistent_cache.clear_cache()
        print("✅ Model caches cleared")
        
    except Exception as e:
        print(f"❌ Error clearing caches: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Warm up chatbot models")
    parser.add_argument("--clear-cache", action="store_true", help="Clear model caches before warming up")
    parser.add_argument("--test-only", action="store_true", help="Only run performance tests")
    parser.add_argument("--vector-store-only", action="store_true", help="Only warm up vector store")
    parser.add_argument("--llm-only", action="store_true", help="Only warm up LLM")
    
    args = parser.parse_args()
    
    print("🚀 Chatbot Model Warmup")
    print("=" * 40)
    
    # Clear caches if requested
    if args.clear_cache:
        clear_caches()
    
    # Set up environment
    optimize_environment()
    setup_cache_directories()
    
    if args.test_only:
        # Only run performance tests
        test_performance()
        return
    
    total_start = time.time()
    success_count = 0
    total_tasks = 0
    
    # Warm up components based on arguments
    if not args.llm_only:
        total_tasks += 1
        if warm_up_vector_store():
            success_count += 1
    
    if not args.vector_store_only:
        total_tasks += 1
        if warm_up_llm():
            success_count += 1
    
    # Test performance
    total_tasks += 1
    if test_performance():
        success_count += 1
    
    total_time = time.time() - total_start
    
    print("=" * 40)
    print(f"🎉 Warmup completed in {total_time:.2f}s")
    print(f"✅ {success_count}/{total_tasks} tasks completed successfully")
    
    if success_count == total_tasks:
        print("🚀 All models are ready! First request should be fast.")
    else:
        print("⚠️  Some components failed to warm up. Check the errors above.")

if __name__ == "__main__":
    main()
