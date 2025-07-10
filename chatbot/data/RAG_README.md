# Enhanced RAG System with Bitext Customer Service Dataset

This implementation integrates the Bitext Customer Service Tagged Training Dataset into your existing chatbot system, providing advanced RAG (Retrieval-Augmented Generation) capabilities for customer service interactions.

## 🚀 Overview

The enhanced RAG system provides:

- **Multi-source vector search**: Combines product data with customer service knowledge
- **Intent-based retrieval**: Leverages the 27 intents from Bitext dataset
- **Category-aware responses**: Handles 10 customer service categories
- **Language variation support**: Processes different linguistic styles and formality levels
- **Contextual response generation**: Provides relevant examples and response patterns

## 📊 Bitext Dataset Integration

### Dataset Structure
- **26,872 question/answer pairs** across customer service scenarios
- **27 intents** organized into 10 categories
- **30+ entity types** for enhanced context understanding
- **12 language variation tags** for handling different communication styles

### Supported Categories
- **ACCOUNT**: create_account, delete_account, edit_account, switch_account
- **ORDER**: cancel_order, change_order, place_order
- **PAYMENT**: check_payment_methods, payment_issue
- **REFUND**: check_refund_policy, track_refund
- **DELIVERY**: delivery_options
- **SHIPPING_ADDRESS**: change_shipping_address, set_up_shipping_address
- **INVOICE**: check_invoice, get_invoice
- **CANCELLATION_FEE**: check_cancellation_fee
- **NEWSLETTER**: newsletter_subscription
- **FEEDBACK**: complaint, review

## 🛠️ Installation & Setup

### 1. Quick Setup
```bash
# Run the setup script
python setup_rag_system.py
```

### 2. Manual Setup

#### Install Dependencies
```bash
pip install faiss-cpu pandas numpy requests langchain langchain-community python-dotenv
```

#### Configure Environment
Create or update `.env` file:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DATABASE_URL=sqlite:///./chatbot.db
```

#### Download Dataset
1. Download the Bitext dataset from [HuggingFace](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
2. Save as CSV, JSON, or Parquet format

#### Process Dataset
```bash
# Process the Bitext dataset for RAG
python data/process_bitext_dataset.py
```

## 🎯 Usage

### Basic RAG Operations

```python
from agent.customer_service_rag import customer_service_rag

# Analyze user query context
context = customer_service_rag.analyze_query_context("I want to cancel my order")
print(context)
# Output: {'query_type': 'order', 'confidence': 0.8, 'suggested_category': 'ORDER'}

# Get relevant examples
examples = customer_service_rag.get_relevant_examples("How do I return an item?", k=3)
for example in examples:
    print(f"Intent: {example['intent']}")
    print(f"Response: {example['response']}")

# Generate contextual response
response = customer_service_rag.generate_contextual_response(
    "I need help with my payment", 
    intent="payment_issue"
)
print(response['suggested_approach'])  # 'payment_assistance'
```

### Enhanced Agent Integration

The system automatically enhances your existing agent with:

```python
# Enhanced RAG examples in intent classification
def get_enhanced_rag_examples(user_input, seller_id, k=3):
    # Combines customer service + product examples
    cs_context = customer_service_rag.generate_contextual_response(user_input)
    # Returns contextual examples for better intent recognition
```

### Vector Store Operations

```python
from vector_store.vector_store import vector_store

# Get dataset information
info = vector_store.get_dataset_info()
print(f"Total documents: {info['total_documents']}")
print(f"Available intents: {info['available_intents']}")

# Search by intent
results = vector_store.search_by_intent("cancel_order", k=3)

# Search by category
results = vector_store.search_by_category("PAYMENT", k=3)

# General similarity search
results = vector_store.similarity_search("I need help with billing", k=5)
```

## 🧪 Testing

### Run Demo Script
```bash
python demo_rag_system.py
```

The demo includes:
- Vector store functionality testing
- RAG agent capability testing
- Full contextual response generation
- Interactive testing mode

### Test Coverage
- ✅ Dataset loading and validation
- ✅ Embedding generation and storage
- ✅ Similarity search functionality
- ✅ Intent and category classification
- ✅ Contextual response generation
- ✅ Multi-source vector search

## 📁 File Structure

```
chatbot/
├── agent/
│   ├── agent.py                    # Enhanced main agent
│   └── customer_service_rag.py     # RAG agent for customer service
├── data/
│   ├── process_bitext_dataset.py   # Dataset processing script
│   ├── bitext_embeddings.pkl       # Processed embeddings (generated)
│   └── sample_customer_service_dataset.json  # Sample data
├── vector_store/
│   ├── vector_store.py            # Enhanced vector store
│   └── generate_embeddings.py     # Embedding generation utilities
├── demo_rag_system.py             # Demo and testing script
├── setup_rag_system.py            # Setup automation script
└── requirements.txt               # Dependencies
```

## 🔧 Configuration

### Vector Store Settings
- **Embedding dimension**: 768 (DeepSeek default)
- **Search algorithm**: FAISS IndexFlatL2
- **Default k**: 3-5 results per query
- **Fallback strategy**: Keyword-based search

### RAG Agent Settings
- **Context window**: Configurable per query
- **Intent confidence threshold**: 0.7
- **Language variation support**: All Bitext flags
- **Response strategy**: Context-adaptive

## 🎨 Customization

### Adding Custom Intents
```python
# Extend the intent mapping
custom_intents = {
    "custom_intent": "CUSTOM_CATEGORY",
    # Add your intents here
}
customer_service_rag.intent_mapping.update(custom_intents)
```

### Custom Response Patterns
```python
# Override response analysis
def custom_response_analysis(examples):
    # Your custom logic
    return {"pattern": "custom"}

customer_service_rag._analyze_response_patterns = custom_response_analysis
```

### Dataset Integration
```python
# Add your own dataset
processor = BitextDatasetProcessor()
df = processor.load_dataset("your_dataset.csv")
processed = processor.process_dataset(df)
processor.save_processed_data("data")
```

## 🚨 Troubleshooting

### Common Issues

1. **"Bitext dataset not found"**
   - Run `python data/process_bitext_dataset.py`
   - Ensure dataset file path is correct

2. **"No API key found"**
   - Check `.env` file configuration
   - Verify `DEEPSEEK_API_KEY` is set

3. **"Empty search results"**
   - Check if embeddings are generated
   - Verify dataset processing completed successfully

4. **"Import errors"**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for RAG operations
logger = logging.getLogger("agent.customer_service_rag")
logger.setLevel(logging.DEBUG)
```

## 📈 Performance Optimization

### Embedding Caching
- Embeddings are cached in pickle files
- Regenerate only when dataset changes
- Use batch processing for large datasets

### Search Optimization
- Use intent/category filters for faster search
- Implement query preprocessing
- Cache frequent queries

### Memory Management
- Load embeddings on-demand for large datasets
- Use embedding compression techniques
- Implement lazy loading for metadata

## 🤝 Contributing

1. **Test your changes** with `demo_rag_system.py`
2. **Follow logging conventions** for debugging
3. **Update documentation** for new features
4. **Maintain backward compatibility** with existing agents

## 📄 License

This implementation respects the Bitext dataset license (CDLA Sharing 1.0) and is designed for educational and commercial use in customer service applications.

## 🔗 Resources

- [Bitext Dataset on HuggingFace](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [Bitext Chatbot Verticals](https://www.bitext.com/chatbot-verticals/)
- [DeepSeek API Documentation](https://platform.deepseek.com/api-docs/)
- [FAISS Documentation](https://faiss.ai/cpp_api/)

---

🎉 **Your enhanced RAG system is ready to provide intelligent customer service responses!**
