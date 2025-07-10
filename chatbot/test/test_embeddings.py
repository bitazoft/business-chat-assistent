"""
Simple test embeddings generator
Creates basic embeddings for testing the RAG system without external dependencies
"""

import numpy as np
import json
import os
import pickle
from typing import List, Dict, Any
import hashlib

class SimpleEmbeddingGenerator:
    """
    Simple embedding generator for testing purposes
    Uses word frequency and hashing to create consistent embeddings
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
        self.vocab = set()
        
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to a simple but consistent embedding vector"""
        # Clean and tokenize
        words = text.lower().replace('\n', ' ').split()
        
        # Create base vector
        vector = np.zeros(self.embedding_dim)
        
        if not words:
            return vector
        
        # Simple approach: hash each word to positions in the vector
        for i, word in enumerate(words):
            # Use hash to get consistent positions for each word
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            
            # Map to multiple positions in the vector for better representation
            for j in range(3):  # Use 3 positions per word
                pos = (word_hash + j * 1000) % self.embedding_dim
                # Weight by position in text (earlier words get higher weight)
                weight = 1.0 / (i + 1)
                vector[pos] += weight
        
        # Add some semantic features based on text characteristics
        text_len = len(text)
        vector[0] = text_len / 1000.0  # Length feature
        
        # Question words get special treatment
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'can', 'do', 'is', 'are']
        for qword in question_words:
            if qword in text.lower():
                pos = (hash(qword) % (self.embedding_dim - 10)) + 5
                vector[pos] += 2.0
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts to embeddings"""
        embeddings = []
        for text in texts:
            embedding = self._text_to_vector(text)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding"""
        return self._text_to_vector(text).astype(np.float32)

def create_test_bitext_dataset():
    """Create a test dataset similar to Bitext format"""
    test_data = [
        {
            "instruction": "How can I cancel my order?",
            "response": "I can help you cancel your order. Please provide your order number and I'll process the cancellation for you.",
            "category": "ORDER",
            "intent": "cancel_order",
            "flags": "B"
        },
        {
            "instruction": "I want to change my shipping address",
            "response": "I can help you update your shipping address. Please provide your order number and the new address details.",
            "category": "SHIPPING_ADDRESS",
            "intent": "change_shipping_address",
            "flags": "B"
        },
        {
            "instruction": "What payment methods do you accept?",
            "response": "We accept major credit cards (Visa, MasterCard, American Express), PayPal, and bank transfers.",
            "category": "PAYMENT",
            "intent": "check_payment_methods",
            "flags": "B"
        },
        {
            "instruction": "How do I create an account?",
            "response": "To create an account, click on 'Sign Up' and fill in your email, password, and basic information. You'll receive a confirmation email to activate your account.",
            "category": "ACCOUNT",
            "intent": "create_account",
            "flags": "B"
        },
        {
            "instruction": "I need to track my refund",
            "response": "I can help you track your refund status. Please provide your order number or refund reference number.",
            "category": "REFUND",
            "intent": "track_refund",
            "flags": "B"
        },
        {
            "instruction": "can u help me with my order?",
            "response": "Of course! I'd be happy to help you with your order. What specific assistance do you need?",
            "category": "ORDER",
            "intent": "general_order_inquiry",
            "flags": "Q"
        },
        {
            "instruction": "I forgot my password",
            "response": "I can help you reset your password. Please click on 'Forgot Password' on the login page and follow the instructions.",
            "category": "ACCOUNT",
            "intent": "recover_password",
            "flags": "B"
        },
        {
            "instruction": "Where is my delivery?",
            "response": "Let me help you track your delivery. Please provide your order number so I can check the current status and location.",
            "category": "DELIVERY",
            "intent": "track_delivery",
            "flags": "B"
        },
        {
            "instruction": "What is your return policy?",
            "response": "Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags attached.",
            "category": "REFUND",
            "intent": "check_refund_policy",
            "flags": "B"
        },
        {
            "instruction": "How long does shipping take?",
            "response": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. You'll receive tracking information once your order ships.",
            "category": "DELIVERY",
            "intent": "delivery_options",
            "flags": "B"
        },
        {
            "instruction": "I have a complaint about my order",
            "response": "I'm sorry to hear about the issue with your order. Please provide your order number and details about the problem so I can assist you.",
            "category": "FEEDBACK",
            "intent": "complaint",
            "flags": "B"
        },
        {
            "instruction": "Can I get an invoice for my purchase?",
            "response": "Absolutely! I can help you get an invoice for your purchase. Please provide your order number and I'll generate the invoice for you.",
            "category": "INVOICE",
            "intent": "get_invoice",
            "flags": "B"
        },
        {
            "instruction": "How do I subscribe to your newsletter?",
            "response": "You can subscribe to our newsletter by entering your email address in the newsletter signup form at the bottom of our website.",
            "category": "NEWSLETTER",
            "intent": "newsletter_subscription",
            "flags": "B"
        },
        {
            "instruction": "Is there a cancellation fee?",
            "response": "Cancellation fees depend on when you cancel. Orders cancelled within 24 hours have no fee. After that, a small processing fee may apply.",
            "category": "CANCELLATION_FEE",
            "intent": "check_cancellation_fee",
            "flags": "B"
        },
        {
            "instruction": "help me plz!!!",
            "response": "Of course! I'm here to help you. Could you please tell me what specific assistance you need?",
            "category": "GENERAL",
            "intent": "general_inquiry",
            "flags": "Q"
        }
    ]
    
    return test_data

def generate_test_embeddings():
    """Generate test embeddings for the Bitext-style dataset"""
    print("🚀 Generating Test Embeddings for RAG System")
    print("=" * 50)
    
    # Create test dataset
    test_data = create_test_bitext_dataset()
    print(f"📊 Created test dataset with {len(test_data)} examples")
    
    # Initialize embedding generator
    embedding_generator = SimpleEmbeddingGenerator(embedding_dim=384)
    
    # Process data into RAG format
    processed_data = []
    texts = []
    metadata = []
    
    for idx, item in enumerate(test_data):
        instruction = item['instruction']
        response = item['response']
        category = item['category']
        intent = item['intent']
        flags = item['flags']
        
        # Create document entry for instruction
        doc_entry = {
            'page_content': instruction,
            'metadata': {
                'response': response,
                'category': category,
                'intent': intent,
                'flags': flags,
                'source': 'bitext_customer_service',
                'doc_id': f"test_bitext_{idx}",
                'use_case': 'customer_service'
            }
        }
        processed_data.append(doc_entry)
        texts.append(instruction)
        metadata.append(doc_entry['metadata'])
        
        # Also add response for reverse lookup
        response_entry = {
            'page_content': response,
            'metadata': {
                'original_instruction': instruction,
                'category': category,
                'intent': intent,
                'flags': flags,
                'source': 'bitext_customer_service_response',
                'doc_id': f"test_bitext_response_{idx}",
                'use_case': 'response_pattern'
            }
        }
        processed_data.append(response_entry)
        texts.append(response)
        metadata.append(response_entry['metadata'])
    
    print(f"✅ Processed {len(processed_data)} documents")
    
    # Generate embeddings
    print("🔄 Generating embeddings...")
    embeddings = embedding_generator.encode(texts)
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    
    # Create output directory
    os.makedirs("data", exist_ok=True)
    
    # Save embeddings
    output_file = "data/bitext_embeddings.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "documents": texts,
            "metadata": metadata,
            "dataset_info": {
                "source": "bitext_customer_service_test",
                "total_docs": len(texts),
                "categories": list(set(m['category'] for m in metadata if m.get('category'))),
                "intents": list(set(m['intent'] for m in metadata if m.get('intent'))),
                "embedding_method": "simple_hash_based",
                "embedding_dim": 384
            }
        }, f)
    
    print(f"✅ Saved embeddings to {output_file}")
    
    # Save processed data as JSON for inspection
    json_file = "data/bitext_processed.json"
    with open(json_file, "w", encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved processed data to {json_file}")
    
    # Display statistics
    categories = list(set(m['category'] for m in metadata if m.get('category')))
    intents = list(set(m['intent'] for m in metadata if m.get('intent')))
    
    print("\n📊 Dataset Statistics:")
    print(f"Total documents: {len(processed_data)}")
    print(f"Categories ({len(categories)}): {categories}")
    print(f"Intents ({len(intents)}): {intents}")
    
    return output_file

def test_embeddings():
    """Test the generated embeddings"""
    print("\n🧪 Testing Generated Embeddings")
    print("=" * 40)
    
    try:
        # Load the embeddings
        with open("data/bitext_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
        
        embeddings = data["embeddings"]
        documents = data["documents"]
        metadata = data["metadata"]
        
        print(f"✅ Loaded {len(documents)} documents")
        print(f"📐 Embedding shape: {embeddings.shape}")
        
        # Test similarity search
        generator = SimpleEmbeddingGenerator(embedding_dim=384)
        
        test_queries = [
            "How to cancel order?",
            "Change shipping address",
            "Payment methods",
            "Create account",
            "Track refund"
        ]
        
        print("\n🔍 Testing similarity search:")
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Generate query embedding
            query_embedding = generator.encode_single(query)
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(embeddings):
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((similarity, i))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Show top 3 results
            for j, (sim, idx) in enumerate(similarities[:3]):
                doc = documents[idx]
                intent = metadata[idx].get('intent', 'N/A')
                print(f"  {j+1}. [{intent}] {doc[:60]}... (similarity: {sim:.3f})")
        
        print("\n✅ Embedding test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

def main():
    """Main function"""
    print("🎯 Test Embeddings Generator")
    print("This creates simple embeddings for testing your RAG system")
    print("=" * 60)
    
    choice = input("\nWhat would you like to do?\n1. Generate test embeddings\n2. Test existing embeddings\n3. Both\nChoose (1-3): ").strip()
    
    if choice in ['1', '3']:
        output_file = generate_test_embeddings()
        print(f"\n🎉 Test embeddings generated successfully!")
        print(f"📁 File: {output_file}")
    
    if choice in ['2', '3']:
        test_embeddings()
    
    print("\n💡 Next steps:")
    print("1. Run: python demo_rag_system.py")
    print("2. Test your RAG system with the generated embeddings")
    print("3. The system will now work without external API dependencies!")

if __name__ == "__main__":
    main()
