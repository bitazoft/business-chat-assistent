"""
Process the Bitext Customer Service dataset for RAG implementation
This script loads, processes, and prepares the dataset for vector store indexing
"""

import pandas as pd
import json
import os
import requests
import numpy as np
import pickle
from typing import List, Dict, Any
from pathlib import Path

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# API Keys for different embedding providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")

# Try to import transformers for Hugging Face models
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not installed. Install with: pip install transformers torch")

class BitextDatasetProcessor:
    """Process Bitext Customer Service dataset for RAG system"""
    
    def __init__(self, dataset_path: str = None, embedding_model: str = "sentence-transformers"):
        self.dataset_path = dataset_path
        self.processed_data = []
        self.embeddings = None
        self.embedding_model = embedding_model
        self.embedding_dimension = 384  # Default dimension, will be updated based on model
        
        # Initialize embedding model
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on the specified type"""
        print(f"🔧 Initializing embedding model: {self.embedding_model}")
        
        if self.embedding_model == "sentence-transformers":
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    # Use a good multilingual model for customer service
                    self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.embedding_dimension = self.st_model.get_sentence_embedding_dimension()
                    print(f"✅ Loaded SentenceTransformer model with dimension {self.embedding_dimension}")
                except Exception as e:
                    print(f"❌ Error loading SentenceTransformer: {e}")
                    self.embedding_model = "openai"  # Fallback
            else:
                print("❌ sentence-transformers not available, falling back to OpenAI")
                self.embedding_model = "openai"
        
        if self.embedding_model == "openai":
            if not OPENAI_API_KEY:
                print("❌ OpenAI API key not found, falling back to huggingface")
                self.embedding_model = "huggingface"
            else:
                self.embedding_dimension = 1536  # OpenAI text-embedding-3-small
                print("✅ OpenAI embeddings configured")
        
        if self.embedding_model == "huggingface":
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Use a lightweight BERT model
                    self.hf_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                    self.hf_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                    self.embedding_dimension = 384
                    print(f"✅ Loaded Hugging Face model with dimension {self.embedding_dimension}")
                except Exception as e:
                    print(f"❌ Error loading Hugging Face model: {e}")
                    self.embedding_model = "mock"
            else:
                print("❌ transformers not available, using mock embeddings")
                self.embedding_model = "mock"
        
        if self.embedding_model == "mock":
            self.embedding_dimension = 384  # Standard dimension for mock
            print("⚠️ Using mock embeddings (random vectors) - not suitable for production")
        
        print(f"🎯 Final embedding model: {self.embedding_model}, dimension: {self.embedding_dimension}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using the specified model"""
        print(f"🔄 Generating embeddings for {len(texts)} texts using {self.embedding_model}...")
        
        if self.embedding_model == "sentence-transformers":
            return self._generate_sentence_transformer_embeddings(texts)
        elif self.embedding_model == "openai":
            return self._generate_openai_embeddings(texts)
        elif self.embedding_model == "huggingface":
            return self._generate_huggingface_embeddings(texts)
        elif self.embedding_model == "mock":
            return self._generate_mock_embeddings(texts)
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
    
    def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using SentenceTransformers"""
        embeddings = []
        batch_size = 32  # Process in batches for efficiency
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                # Clean texts
                clean_batch = [text.replace('\n', ' ').strip()[:512] for text in batch]  # Limit length
                batch_embeddings = self.st_model.encode(clean_batch, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"❌ Error in batch {i//batch_size + 1}: {e}")
                # Add zero vectors for failed batch
                for _ in batch:
                    embeddings.append(np.zeros(self.embedding_dimension))
        
        return np.array(embeddings, dtype=np.float32)
    
    def _generate_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        embeddings = []
        batch_size = 100  # OpenAI allows larger batches
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"📦 Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                # Clean and prepare batch
                clean_batch = [text.replace('\n', ' ').strip()[:8000] for text in batch]
                
                payload = {
                    "input": clean_batch,
                    "model": "text-embedding-3-small"  # Cost-effective model
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    embeddings.extend(batch_embeddings)
                else:
                    print(f"⚠️ API error: {response.status_code} - {response.text}")
                    # Add zero vectors for failed batch
                    for _ in batch:
                        embeddings.append(np.zeros(self.embedding_dimension).tolist())
                        
            except Exception as e:
                print(f"❌ Error in batch {i//batch_size + 1}: {e}")
                for _ in batch:
                    embeddings.append(np.zeros(self.embedding_dimension).tolist())
        
        return np.array(embeddings, dtype=np.float32)
    
    def _generate_huggingface_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Hugging Face transformers"""
        embeddings = []
        batch_size = 16  # Smaller batches for transformers
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hf_model.to(device)
        self.hf_model.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"📦 Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                try:
                    # Tokenize and encode
                    clean_batch = [text.replace('\n', ' ').strip()[:512] for text in batch]
                    
                    encoded = self.hf_tokenizer(
                        clean_batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(device)
                    
                    # Get embeddings
                    outputs = self.hf_model(**encoded)
                    # Use mean pooling
                    embeddings_batch = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.extend(embeddings_batch)
                    
                except Exception as e:
                    print(f"❌ Error in batch {i//batch_size + 1}: {e}")
                    for _ in batch:
                        embeddings.append(np.zeros(self.embedding_dimension))
        
        return np.array(embeddings, dtype=np.float32)
    
    def _generate_mock_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings (random vectors) for testing"""
        print("⚠️ Generating mock embeddings - not suitable for production use!")
        
        embeddings = []
        np.random.seed(42)  # For reproducible results
        
        for i, text in enumerate(texts):
            # Create a simple hash-based embedding for consistency
            text_hash = hash(text.lower().strip()) % 1000000
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 1, self.embedding_dimension)
            embeddings.append(embedding)
            
            if (i + 1) % 100 == 0:
                print(f"📦 Generated {i + 1}/{len(texts)} mock embeddings")
        
        return np.array(embeddings, dtype=np.float32)
        
    def load_dataset(self, dataset_path: str = None) -> pd.DataFrame:
        """Load the Bitext dataset from various formats"""
        if dataset_path:
            self.dataset_path = dataset_path
            
        if not self.dataset_path:
            raise ValueError("Dataset path must be provided")
            
        file_ext = Path(self.dataset_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(self.dataset_path)
            elif file_ext == '.json':
                df = pd.read_json(self.dataset_path)
            elif file_ext == '.jsonl':
                df = pd.read_json(self.dataset_path, lines=True)
            elif file_ext == '.parquet':
                df = pd.read_parquet(self.dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            print(f"✅ Successfully loaded dataset with {len(df)} rows")
            print(f"📊 Columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            raise
    
    def validate_dataset_structure(self, df: pd.DataFrame) -> bool:
        """Validate that the dataset has the expected Bitext structure"""
        required_columns = ['instruction', 'response', 'category', 'intent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"📋 Available columns: {list(df.columns)}")
            return False
            
        print("✅ Dataset structure validation passed")
        return True
    
    def process_dataset(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process the dataset into RAG-ready format"""
        processed_data = []
        
        for idx, row in df.iterrows():
            # Extract basic information
            instruction = str(row.get('instruction', '')).strip()
            response = str(row.get('response', '')).strip()
            category = str(row.get('category', '')).strip()
            intent = str(row.get('intent', '')).strip()
            flags = str(row.get('flags', '')).strip()
            
            # Skip empty entries
            if not instruction or not response:
                continue
            
            # Create document entry
            doc_entry = {
                'page_content': instruction,
                'metadata': {
                    'response': response,
                    'category': category,
                    'intent': intent,
                    'flags': flags,
                    'source': 'bitext_customer_service',
                    'doc_id': f"bitext_{idx}",
                    'use_case': 'customer_service'
                }
            }
            
            processed_data.append(doc_entry)
            
            # Also create reverse lookup for response patterns
            if len(response) > 10:  # Only for substantial responses
                response_entry = {
                    'page_content': response,
                    'metadata': {
                        'original_instruction': instruction,
                        'category': category,
                        'intent': intent,
                        'flags': flags,
                        'source': 'bitext_customer_service_response',
                        'doc_id': f"bitext_response_{idx}",
                        'use_case': 'response_pattern'
                    }
                }
                processed_data.append(response_entry)
        
        self.processed_data = processed_data
        print(f"✅ Processed {len(processed_data)} documents for RAG")
        return processed_data
    
    def save_processed_data(self, output_dir: str = "data"):
        """Save processed data and embeddings"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.processed_data:
            raise ValueError("No processed data to save. Run process_dataset first.")
        
        # Extract texts and metadata
        texts = [doc['page_content'] for doc in self.processed_data]
        metadata = [doc['metadata'] for doc in self.processed_data]
        
        # Generate embeddings if not already done
        if self.embeddings is None:
            self.embeddings = self.generate_embeddings(texts)
        
        # Save to pickle file
        output_file = os.path.join(output_dir, "bitext_embeddings.pkl")
        with open(output_file, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "documents": texts,
                "metadata": metadata,
                "dataset_info": {
                    "source": "bitext_customer_service",
                    "total_docs": len(texts),
                    "categories": list(set(m['category'] for m in metadata if m.get('category'))),
                    "intents": list(set(m['intent'] for m in metadata if m.get('intent')))
                }
            }, f)
        
        print(f"✅ Saved embeddings to {output_file}")
        
        # Also save as JSON for inspection
        json_file = os.path.join(output_dir, "bitext_processed.json")
        with open(json_file, "w", encoding='utf-8') as f:
            json.dump(self.processed_data[:100], f, indent=2, ensure_ascii=False)  # Save first 100 for inspection
        
        print(f"✅ Saved sample processed data to {json_file}")
        
        return output_file
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed dataset"""
        if not self.processed_data:
            return {}
        
        metadata_list = [doc['metadata'] for doc in self.processed_data]
        
        categories = [m.get('category', '') for m in metadata_list if m.get('category')]
        intents = [m.get('intent', '') for m in metadata_list if m.get('intent')]
        flags = [m.get('flags', '') for m in metadata_list if m.get('flags')]
        
        stats = {
            "total_documents": len(self.processed_data),
            "unique_categories": len(set(categories)),
            "unique_intents": len(set(intents)),
            "categories": list(set(categories)),
            "intents": list(set(intents)),
            "sample_flags": list(set(flag for flag in flags if flag))[:10]
        }
        
        return stats

def main():
    """Main function to process Bitext dataset"""
    print("🚀 Starting Bitext Customer Service Dataset Processing")
    print("📋 Available embedding models:")
    print("  1. sentence-transformers (Recommended - Free, Local)")
    print("  2. openai (Requires API key)")
    print("  3. huggingface (Free, Local - Requires transformers)")
    print("  4. mock (Testing only - Random vectors)")
    
    # Choose embedding model
    model_choice = input("\n🤖 Choose embedding model (1-4) [1]: ").strip() or "1"
    
    model_map = {
        "1": "sentence-transformers",
        "2": "openai", 
        "3": "huggingface",
        "4": "mock"
    }
    
    embedding_model = model_map.get(model_choice, "sentence-transformers")
    print(f"✅ Selected: {embedding_model}")
    
    # Initialize processor
    processor = BitextDatasetProcessor(embedding_model=embedding_model)
    
    # You'll need to provide the path to your downloaded Bitext dataset
    # For example: "path/to/bitext_customer_service_dataset.csv"
    dataset_path = input("\n📁 Enter the path to your Bitext dataset file: ").strip()
    
    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Invalid dataset path. Please download the dataset first.")
        print("💡 You can download it from: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        
        # Offer to use sample data
        use_sample = input("\n🧪 Use sample data for testing? (y/n): ").strip().lower()
        if use_sample in ['y', 'yes']:
            sample_data = create_sample_dataset()
            df = pd.DataFrame(sample_data)
            print("✅ Using sample dataset for testing")
        else:
            return
    else:
        try:
            # Load and validate dataset
            df = processor.load_dataset(dataset_path)
            
            if not processor.validate_dataset_structure(df):
                return
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return
    
    try:
        # Process dataset
        processed_data = processor.process_dataset(df)
        
        # Show statistics
        stats = processor.get_dataset_stats()
        print("\n📊 Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"{key}: {value[:5]}... (showing first 5)")
            else:
                print(f"{key}: {value}")
        
        # Save processed data
        output_file = processor.save_processed_data()
        
        print(f"\n🎉 Successfully processed Bitext dataset!")
        print(f"📁 Output file: {output_file}")
        print(f"📈 Total documents: {len(processed_data)}")
        print(f"🔧 Embedding model used: {embedding_model}")
        print(f"📐 Embedding dimension: {processor.embedding_dimension}")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {str(e)}")
        raise

def create_sample_dataset():
    """Create a sample dataset for testing"""
    return [
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
            "response": "To create an account, click on 'Sign Up' and fill in your email, password, and basic information.",
            "category": "ACCOUNT", 
            "intent": "create_account",
            "flags": "B"
        },
        {
            "instruction": "I need to track my refund",
            "response": "I can help you track your refund status. Please provide your order number or refund reference.",
            "category": "REFUND",
            "intent": "track_refund",
            "flags": "B"
        },
        {
            "instruction": "can u help me with my order?",
            "response": "Of course! I'd be happy to help you with your order. What specific assistance do you need?",
            "category": "ORDER",
            "intent": "general_order_inquiry", 
            "flags": "Q"  # Colloquial
        },
        {
            "instruction": "I forgot my password",
            "response": "I can help you reset your password. Please click on 'Forgot Password' on the login page.",
            "category": "ACCOUNT",
            "intent": "recover_password",
            "flags": "B"
        },
        {
            "instruction": "Where is my delivery?",
            "response": "Let me help you track your delivery. Please provide your order number so I can check the status.",
            "category": "DELIVERY",
            "intent": "track_delivery",
            "flags": "B"
        }
    ]

if __name__ == "__main__":
    main()
