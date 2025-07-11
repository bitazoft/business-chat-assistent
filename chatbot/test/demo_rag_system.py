"""
Demo script to test the enhanced RAG system with Bitext dataset
Run this to verify that your RAG implementation is working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.customer_service_rag import customer_service_rag
from vector_store.vector_store import vector_store
from utils.logger import get_logger
import json

logger = get_logger(__name__)

def test_vector_store():
    """Test the vector store functionality"""
    print("🧪 Testing Vector Store...")
    
    # Test basic info
    info = vector_store.get_dataset_info()
    print(f"📊 Dataset Info: {json.dumps(info, indent=2)}")
    
    # Test similarity search
    test_queries = [
        "How can I cancel my order?",
        "I want to change my shipping address", 
        "What are your payment methods?",
        "I need to create an account",
        "Track my refund status"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        results = vector_store.similarity_search(query, k=3)
        
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Content: {result.page_content[:100]}...")
            print(f"    Intent: {result.metadata.get('intent', 'N/A')}")
            print(f"    Category: {result.metadata.get('category', 'N/A')}")

def test_rag_agent():
    """Test the RAG agent functionality"""
    print("\n🤖 Testing RAG Agent...")
    
    # Test context analysis
    test_queries = [
        "I need help with my order",
        "How do I create a new account?",
        "What payment methods do you accept?",
        "I want to return this item",
        "My delivery is late"
    ]
    
    for query in test_queries:
        print(f"\n📝 Analyzing query: '{query}'")
        
        # Test context analysis
        context = customer_service_rag.analyze_query_context(query)
        print(f"  Context: {json.dumps(context, indent=4)}")
        
        # Test getting relevant examples
        examples = customer_service_rag.get_relevant_examples(query, k=3)
        print(f"  Found {len(examples)} relevant examples:")
        
        for i, example in enumerate(examples[:2]):  # Show first 2
            print(f"    Example {i+1}:")
            print(f"      Intent: {example.get('intent', 'N/A')}")
            print(f"      Instruction: {example.get('instruction', '')[:80]}...")
            print(f"      Response: {example.get('response', '')[:80]}...")

def test_full_contextual_response():
    """Test full contextual response generation"""
    print("\n🎯 Testing Full Contextual Response...")
    
    test_cases = [
        {
            "query": "I want to cancel my recent order",
            "intent": "cancel_order"
        },
        {
            "query": "How can I change my billing address?", 
            "intent": "change_shipping_address"
        },
        {
            "query": "I forgot my password",
            "intent": "account"
        }
    ]
    
    for case in test_cases:
        print(f"\n🔄 Processing: '{case['query']}'")
        
        response = customer_service_rag.generate_contextual_response(
            case["query"], 
            case.get("intent")
        )
        
        print(f"  Examples found: {len(response.get('examples', []))}")
        print(f"  Response patterns: {response.get('response_patterns', {})}")
        print(f"  Suggested approach: {response.get('suggested_approach', 'N/A')}")
        
        # Show one example
        examples = response.get('examples', [])
        if examples:
            ex = examples[0]
            print(f"  Top example:")
            print(f"    Intent: {ex.get('intent', 'N/A')}")
            print(f"    Category: {ex.get('category', 'N/A')}")
            print(f"    Response sample: {ex.get('response', '')[:100]}...")

def check_dataset_availability():
    """Check if the Bitext dataset is properly loaded"""
    print("📋 Checking Dataset Availability...")
    
    # Check available intents
    intents = customer_service_rag.get_available_intents()
    print(f"Available intents ({len(intents)}): {intents[:10]}...")  # Show first 10
    
    # Check available categories
    categories = customer_service_rag.get_available_categories()
    print(f"Available categories ({len(categories)}): {categories}")
    
    # Check if we have Bitext data
    info = customer_service_rag.get_dataset_info()
    has_bitext = any('bitext' in str(source).lower() for source in info.get('sources', []))
    
    if has_bitext:
        print("✅ Bitext dataset is loaded and available!")
    else:
        print("⚠️ Bitext dataset not found. Please run the data processing script first.")
        print("💡 Steps to load Bitext dataset:")
        print("   1. Download the dataset from HuggingFace")
        print("   2. Run: python data/process_bitext_dataset.py")

def interactive_test():
    """Interactive testing mode"""
    print("\n🎮 Interactive Testing Mode")
    print("Enter queries to test the RAG system (type 'quit' to exit):")
    
    while True:
        try:
            query = input("\n👤 Your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not query:
                continue
                
            print("🔍 Processing...")
            
            # Get contextual response
            response = customer_service_rag.generate_contextual_response(query)
            
            print(f"\n📊 Analysis Results:")
            print(f"  Query type: {response['context_analysis'].get('query_type', 'N/A')}")
            print(f"  Confidence: {response['context_analysis'].get('confidence', 0):.2f}")
            print(f"  Suggested approach: {response.get('suggested_approach', 'N/A')}")
            
            examples = response.get('examples', [])
            if examples:
                print(f"\n📚 Top relevant example:")
                ex = examples[0]
                print(f"  Intent: {ex.get('intent', 'N/A')}")
                print(f"  Category: {ex.get('category', 'N/A')}")
                print(f"  Sample response: {ex.get('response', 'N/A')}")
            else:
                print("\n❌ No relevant examples found")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def main():
    """Main demo function"""
    print("🚀 Enhanced RAG System Demo")
    print("=" * 50)
    
    try:
        # Check dataset availability first
        check_dataset_availability()
        
        # Run tests
        test_vector_store()
        test_rag_agent()
        test_full_contextual_response()
        
        # Ask for interactive mode
        choice = input("\n🎮 Would you like to try interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_test()
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure you have processed the Bitext dataset")
        print("2. Check your API keys in .env file")
        print("3. Verify all dependencies are installed")

if __name__ == "__main__":
    main()
