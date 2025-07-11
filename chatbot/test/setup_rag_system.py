"""
Setup script for the Enhanced RAG System with Bitext Dataset
This script helps you set up and configure the RAG system
"""

import os
import sys
import urllib.request
import zipfile
import json
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'faiss-cpu',
        'pandas',
        'numpy',
        'requests',
        'langchain',
        'langchain-community',
        'langchain-openai',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are installed!")
    return True

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "data",
        "logs",
        "vector_store"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")
    
    print("✅ Directories created!")

def setup_env_file():
    """Set up environment file"""
    print("\n⚙️ Setting up environment configuration...")
    
    env_file = ".env"
    
    if os.path.exists(env_file):
        print(f"  📋 {env_file} already exists")
        with open(env_file, 'r') as f:
            content = f.read()
            
        if "DEEPSEEK_API_KEY" not in content:
            print("  ⚠️ DEEPSEEK_API_KEY not found in .env")
            add_keys = input("  Add API keys to .env? (y/n): ").strip().lower()
            if add_keys in ['y', 'yes']:
                append_env_keys(env_file)
        else:
            print("  ✅ Environment file is properly configured")
    else:
        print(f"  📝 Creating {env_file}")
        create_env_file(env_file)
    
def create_env_file(env_file):
    """Create a new .env file"""
    env_content = """# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# Database Configuration
DATABASE_URL=sqlite:///./chatbot.db

# Optional: OpenAI API for fallback
# OPENAI_API_KEY=your_openai_api_key_here

# Logging Configuration
LOG_LEVEL=INFO
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"  ✅ Created {env_file}")
    print("  ⚠️ Please edit .env and add your actual API keys!")

def append_env_keys(env_file):
    """Append missing API keys to existing .env file"""
    with open(env_file, 'a') as f:
        f.write("""
# DeepSeek API Configuration (added by setup)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
""")
    print("  ✅ Added DeepSeek API configuration to .env")

def download_sample_dataset():
    """Download a sample dataset for testing"""
    print("\n📥 Would you like to download a sample customer service dataset?")
    choice = input("This will help you test the system (y/n): ").strip().lower()
    
    if choice not in ['y', 'yes']:
        return
    
    print("📦 Creating sample dataset...")
    
    # Create a sample dataset based on Bitext structure
    sample_data = [
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
        }
    ]
    
    # Save as JSON
    sample_file = "data/sample_customer_service_dataset.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Created sample dataset: {sample_file}")
    print("  💡 You can use this file to test the data processing script")

def provide_next_steps():
    """Provide next steps to the user"""
    print("\n🎯 Next Steps:")
    print("=" * 50)
    
    print("\n1. 📝 Configure your API keys:")
    print("   - Edit .env file and add your DeepSeek API key")
    print("   - Get API key from: https://platform.deepseek.com/")
    
    print("\n2. 📊 Prepare your dataset:")
    print("   - Download Bitext dataset from HuggingFace:")
    print("     https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    print("   - Or use the sample dataset we created")
    
    print("\n3. 🔄 Process the dataset:")
    print("   - Run: python data/process_bitext_dataset.py")
    print("   - This will create embeddings for the RAG system")
    
    print("\n4. 🧪 Test the system:")
    print("   - Run: python demo_rag_system.py")
    print("   - This will test all RAG functionality")
    
    print("\n5. 🚀 Start using the enhanced chatbot:")
    print("   - Your chatbot now has enhanced customer service capabilities!")
    print("   - The RAG system will provide contextual responses based on the dataset")
    
    print("\n📚 Key Features Added:")
    print("  ✅ Customer service intent recognition")
    print("  ✅ Contextual response generation")
    print("  ✅ Multi-source vector search (products + customer service)")
    print("  ✅ Language variation handling")
    print("  ✅ Category-based response routing")

def main():
    """Main setup function"""
    print("🚀 Enhanced RAG System Setup")
    print("=" * 50)
    print("This script will help you set up the RAG system with Bitext dataset support")
    
    try:
        # Check requirements
        if not check_requirements():
            print("\n❌ Please install missing packages first")
            return
        
        # Setup directories
        setup_directories()
        
        # Setup environment
        setup_env_file()
        
        # Offer sample dataset
        download_sample_dataset()
        
        # Provide next steps
        provide_next_steps()
        
        print("\n✅ Setup completed successfully!")
        print("🎉 Your enhanced RAG system is ready to configure!")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()
