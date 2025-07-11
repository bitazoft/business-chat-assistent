#!/usr/bin/env python3
"""
Test script to verify the RAG-based intent retrieval functionality
"""

import sys
import os

# Add the chatbot directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.agent import get_available_intents_from_rag
from utils.logger import get_logger

logger = get_logger(__name__)

def test_rag_intent_retrieval():
    """Test the RAG-based intent retrieval function"""
    
    print("=" * 60)
    print("Testing RAG-based Intent Retrieval")
    print("=" * 60)
    
    # Test with a sample seller ID
    test_seller_id = "123"
    
    try:
        print(f"\n🔍 Testing intent retrieval for seller_id: {test_seller_id}")
        intents = get_available_intents_from_rag(test_seller_id)
        
        print(f"✅ Successfully retrieved intents:")
        print(f"📋 Available intents: {intents}")
        
        # Parse the intents
        intent_list = [intent.strip() for intent in intents.split(",")]
        print(f"📊 Total number of intents: {len(intent_list)}")
        
        # Check if essential intents are present
        essential_intents = ["product_info", "order_tracking", "place_order", "user_management", "general_inquiry"]
        missing_essential = [intent for intent in essential_intents if intent not in intent_list]
        
        if missing_essential:
            print(f"⚠️  Missing essential intents: {missing_essential}")
        else:
            print("✅ All essential intents are present")
        
        # Test with different seller ID
        test_seller_id_2 = "456"
        print(f"\n🔍 Testing intent retrieval for seller_id: {test_seller_id_2}")
        intents_2 = get_available_intents_from_rag(test_seller_id_2)
        print(f"📋 Available intents: {intents_2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_rag_intent_retrieval()
    if success:
        print("\n🎉 RAG-based intent retrieval test completed successfully!")
    else:
        print("\n❌ RAG-based intent retrieval test failed!")
        sys.exit(1)
