#!/usr/bin/env python3
"""
Quick performance test for the optimized vector store
"""

import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_vector_store_performance():
    """Test vector store search performance"""
    print("🔄 Testing optimized vector store performance...")
    
    try:
        from vector_store.vector_store import fast_vector_store
        
        # Test queries
        test_queries = [
            "What products do you have?",
            "How can I track my order?",
            "What is your return policy?",
            "Do you offer international shipping?",
            "How can I contact customer service?"
        ]
        
        print(f"Testing {len(test_queries)} queries...")
        print("-" * 50)
        
        total_time = 0
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            results = fast_vector_store.similarity_search(query, k=3)
            search_time = time.time() - start_time
            total_time += search_time
            
            print(f"{i}. Query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
            print(f"   Time: {search_time:.3f}s | Results: {len(results)}")
        
        print("-" * 50)
        avg_time = total_time / len(test_queries)
        print(f"📊 Total time: {total_time:.3f}s")
        print(f"📊 Average time per query: {avg_time:.3f}s")
        
        # Performance assessment
        if avg_time < 0.1:
            print("🚀 Excellent performance!")
        elif avg_time < 0.5:
            print("✅ Good performance!")
        elif avg_time < 1.0:
            print("⚠️  Acceptable performance")
        else:
            print("❌ Poor performance - check optimization")
            
        return avg_time
        
    except Exception as e:
        print(f"❌ Error testing performance: {str(e)}")
        return None

def test_model_loading():
    """Test if model is already loaded (cached)"""
    print("🔄 Testing model caching...")
    
    try:
        from vector_store.vector_store import ModelCache
        
        model_cache = ModelCache()
        
        # Test loading time (should be fast if cached)
        start_time = time.time()
        model = model_cache.get_model("all-MiniLM-L6-v2")
        load_time = time.time() - start_time
        
        print(f"⏱️  Model loading time: {load_time:.3f}s")
        
        if load_time < 0.1:
            print("✅ Model is cached in memory!")
        elif load_time < 1.0:
            print("⚠️  Model loaded reasonably fast")
        else:
            print("❌ Model not cached - optimization not working")
            
        return load_time
        
    except Exception as e:
        print(f"❌ Error testing model caching: {str(e)}")
        return None

def main():
    print("🧪 Chatbot Performance Test")
    print("=" * 40)
    
    # Test model caching
    model_time = test_model_loading()
    print()
    
    # Test vector store performance
    search_time = test_vector_store_performance()
    
    print("=" * 40)
    if model_time is not None and search_time is not None:
        if model_time < 0.1 and search_time < 0.5:
            print("🎉 Optimization is working well!")
            print("💡 Your chatbot should respond quickly now.")
        else:
            print("⚠️  Performance could be better.")
            print("💡 Try running: python warmup_models.py")
    else:
        print("❌ Performance test failed. Check your setup.")

if __name__ == "__main__":
    main()
