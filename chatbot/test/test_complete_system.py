#!/usr/bin/env python3
"""
Test script to verify the complete multi-agent system with RAG-based intent retrieval
"""

import sys
import os

# Add the chatbot directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.agent import create_multi_agent_system, get_available_intents_from_rag
from utils.logger import get_logger

logger = get_logger(__name__)

def test_multi_agent_with_rag_intents():
    """Test the complete multi-agent system with RAG-based intents"""
    
    print("=" * 60)
    print("Testing Multi-Agent System with RAG-based Intent Retrieval")
    print("=" * 60)
    
    # Test with sample seller and user IDs
    test_seller_id = "123"
    test_user_id = "test_user_001"
    
    try:
        print(f"\n🔧 Creating multi-agent system for seller_id: {test_seller_id}, user_id: {test_user_id}")
        
        # Create the multi-agent system
        multi_agent = create_multi_agent_system(test_seller_id, test_user_id)
        
        print("✅ Multi-agent system created successfully")
        
        # Test different types of queries to see intent detection
        test_queries = [
            "I want to buy a laptop",
            "What products do you have?", 
            "Can you track my order #12345?",
            "I need to update my address",
            "Hello, how can you help me?"
        ]
        
        print(f"\n🔍 Testing intent detection with {len(test_queries)} different queries:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {query} ---")
            
            try:
                # Test the complete flow
                response = multi_agent["executor"]({"input": query})
                print(f"✅ Query processed successfully")
                print(f"📤 Response: {response[:150]}{'...' if len(response) > 150 else ''}")
                
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
                logger.error(f"Query processing failed for '{query}': {str(e)}")
        
        # Test intent retrieval directly
        print(f"\n🔍 Direct intent retrieval test:")
        intents = get_available_intents_from_rag(test_seller_id)
        print(f"📋 Available intents: {intents}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        logger.error(f"Multi-agent test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_multi_agent_with_rag_intents()
    if success:
        print("\n🎉 Multi-agent system with RAG-based intent retrieval test completed!")
    else:
        print("\n❌ Multi-agent system test failed!")
        sys.exit(1)
