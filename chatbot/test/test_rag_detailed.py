#!/usr/bin/env python3
"""
Test script to verify just the RAG-based intent retrieval functionality
without requiring API keys or LLM calls
"""

import sys
import os

# Add the chatbot directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.agent import get_available_intents_from_rag
from utils.logger import get_logger
from vector_store.vector_store import vector_store

logger = get_logger(__name__)

def test_vector_store_intent_data():
    """Test that the vector store contains intent information"""
    
    print("=" * 60)
    print("Testing Vector Store Intent Data")
    print("=" * 60)
    
    try:
        print(f"📊 Total documents in vector store: {len(vector_store.documents)}")
        print(f"📊 Total metadata entries: {len(vector_store.metadata)}")
        
        # Analyze intent distribution
        intent_counts = {}
        category_counts = {}
        
        for metadata in vector_store.metadata:
            intent = metadata.get('intent', 'unknown')
            category = metadata.get('category', 'unknown')
            
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"\n📈 Intent Distribution (top 10):")
        sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for intent, count in sorted_intents:
            print(f"  • {intent}: {count}")
        
        print(f"\n📈 Category Distribution (top 10):")
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for category, count in sorted_categories:
            print(f"  • {category}: {count}")
        
        # Test keyword search for intent-related queries
        print(f"\n🔍 Testing keyword search for intent queries:")
        
        test_queries = [
            "intent classification",
            "order tracking", 
            "product information",
            "account management"
        ]
        
        for query in test_queries:
            results = vector_store.similarity_search(query, k=3)
            print(f"\n  Query: '{query}' - Found {len(results)} results")
            for i, result in enumerate(results[:2]):  # Show top 2
                intent = result.metadata.get('intent', 'N/A')
                category = result.metadata.get('category', 'N/A')
                print(f"    {i+1}. Intent: {intent}, Category: {category}")
                print(f"       Content: {result.page_content[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        logger.error(f"Vector store test failed: {str(e)}")
        return False

def test_rag_intent_retrieval_detailed():
    """Test the RAG intent retrieval with detailed analysis"""
    
    print("\n" + "=" * 60)
    print("Testing Detailed RAG Intent Retrieval")
    print("=" * 60)
    
    test_seller_ids = ["123", "456", "999"]
    
    for seller_id in test_seller_ids:
        print(f"\n🔍 Testing seller_id: {seller_id}")
        
        try:
            intents = get_available_intents_from_rag(seller_id, k=5)
            intent_list = [intent.strip() for intent in intents.split(",")]
            
            print(f"✅ Retrieved {len(intent_list)} intents: {intents}")
            
            # Check for essential intents
            essential_intents = ["product_info", "order_tracking", "place_order", "user_management", "general_inquiry"]
            missing = [intent for intent in essential_intents if intent not in intent_list]
            
            if missing:
                print(f"⚠️  Missing essential intents: {missing}")
            else:
                print("✅ All essential intents present")
            
        except Exception as e:
            print(f"❌ Error for seller_id {seller_id}: {str(e)}")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting comprehensive RAG intent tests...\n")
    
    success1 = test_vector_store_intent_data()
    success2 = test_rag_intent_retrieval_detailed()
    
    if success1 and success2:
        print("\n🎉 All RAG intent tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
