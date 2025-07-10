#!/usr/bin/env python3
"""
Test script to verify vector store connectivity and similarity search
This will help you understand how the vector similarity is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store.vector_store import fast_vector_store
from utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)

def test_vector_connectivity():
    """Test the vector store connectivity and similarity search"""
    
    print("🔍 Testing Vector Store Connectivity...")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "I want to cancel my order",
        "How can I track my package?",
        "What is your return policy?",
        "I need help with billing",
        "Product recommendation"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: Query = '{query}'")
        print("-" * 30)
        
        # Search for similar documents
        results = fast_vector_store.similarity_search(query, k=3)
        
        if not results:
            print("❌ No results found")
            continue
            
        print(f"✅ Found {len(results)} results:")
        
        for j, result in enumerate(results, 1):
            similarity_score = result.metadata.get('similarity_score', 'N/A')
            content_preview = result.page_content[:100] + "..." if len(result.page_content) > 100 else result.page_content
            
            print(f"  {j}. Similarity: {similarity_score}")
            print(f"     Content: {content_preview}")
            print()

def check_embeddings_info():
    """Check information about loaded embeddings"""
    print("\n🔧 Checking Embeddings Information...")
    print("=" * 50)
    
    # Force loading of embeddings
    fast_vector_store._lazy_load()
    
    if fast_vector_store._embeddings is not None and len(fast_vector_store._embeddings) > 0:
        print(f"✅ Embeddings loaded successfully!")
        print(f"   📊 Number of documents: {len(fast_vector_store._documents)}")
        print(f"   📐 Embedding dimension: {fast_vector_store._embeddings.shape[1] if len(fast_vector_store._embeddings.shape) > 1 else 'N/A'}")
        print(f"   📦 Embeddings shape: {fast_vector_store._embeddings.shape}")
        
        # Show sample document
        if len(fast_vector_store._documents) > 0:
            sample_doc = fast_vector_store._documents[0]
            print(f"   📄 Sample document: {str(sample_doc)[:200]}...")
            
    else:
        print("❌ No embeddings loaded!")
        print("   🔍 Checking for embedding files...")
        
        import os
        files_to_check = [
            "data/bitext_embeddings.pkl",
            "data/product_embeddings.pkl"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"   ✅ Found: {file_path}")
            else:
                print(f"   ❌ Missing: {file_path}")

def check_model_availability():
    """Check if sentence-transformers is available"""
    print("\n🤖 Checking Model Availability...")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers is available")
        
        # Try to load the model
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Default embedding model loaded successfully")
            
            # Test encoding
            test_text = "This is a test sentence"
            embedding = model.encode([test_text])
            print(f"✅ Test embedding generated, shape: {embedding.shape}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            
    except ImportError:
        print("❌ sentence-transformers not installed")
        print("   💡 Install with: pip install sentence-transformers")

if __name__ == "__main__":
    print("🚀 Vector Store Connectivity Test")
    print("=" * 50)
    
    # Run all tests
    check_model_availability()
    check_embeddings_info()
    test_vector_connectivity()
    
    print("\n🎯 Summary:")
    print("=" * 50)
    print("This test helps you understand:")
    print("1. Whether embeddings are properly loaded")
    print("2. If vector similarity search is working")
    print("3. What similarity scores look like")
    print("4. Whether sentence-transformers is working")
    print("\nCheck the logs above for any issues!")
