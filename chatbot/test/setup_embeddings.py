"""
Quick setup script for embedding models
Helps you install and test different embedding options
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def test_sentence_transformers():
    """Test sentence-transformers installation"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode(["Hello world"])
        print(f"✅ SentenceTransformers working! Embedding dimension: {test_embedding.shape[1]}")
        return True
    except Exception as e:
        print(f"❌ SentenceTransformers test failed: {e}")
        return False

def test_transformers():
    """Test transformers + torch installation"""
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        inputs = tokenizer("Hello world", return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Transformers working! Model output shape: {outputs.last_hidden_state.shape}")
        return True
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return False

def test_openai():
    """Test OpenAI API (requires API key)"""
    try:
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ OpenAI API key not found in .env file")
            return False
        
        print("✅ OpenAI package installed and API key found")
        print("💡 To test: Run the dataset processing script and choose OpenAI option")
        return True
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Embedding Models Setup")
    print("=" * 40)
    
    print("\n📦 Available embedding options:")
    print("1. SentenceTransformers (Recommended - Free, Local, Easy)")
    print("2. Hugging Face Transformers (Free, Local, More control)")
    print("3. OpenAI (Paid API, High quality)")
    print("4. Install all")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1" or choice == "4":
        print("\n📥 Installing SentenceTransformers...")
        if install_package("sentence-transformers"):
            print("✅ SentenceTransformers installed")
            test_sentence_transformers()
        else:
            print("❌ Failed to install SentenceTransformers")
    
    if choice == "2" or choice == "4":
        print("\n📥 Installing Transformers and PyTorch...")
        if install_package("transformers torch"):
            print("✅ Transformers installed")
            test_transformers()
        else:
            print("❌ Failed to install Transformers")
    
    if choice == "3" or choice == "4":
        print("\n📥 Installing OpenAI...")
        if install_package("openai"):
            print("✅ OpenAI installed")
            test_openai()
        else:
            print("❌ Failed to install OpenAI")
    
    print("\n🎯 Setup Summary:")
    print("=" * 40)
    
    # Test what's available
    working_models = []
    
    print("\n🧪 Testing available models...")
    
    if test_sentence_transformers():
        working_models.append("SentenceTransformers")
    
    if test_transformers():
        working_models.append("Transformers")
    
    if test_openai():
        working_models.append("OpenAI")
    
    if working_models:
        print(f"\n✅ Working models: {', '.join(working_models)}")
        print("\n🚀 You're ready to process the Bitext dataset!")
        print("Run: python data/process_bitext_dataset.py")
    else:
        print("\n❌ No embedding models are working properly")
        print("💡 Try installing sentence-transformers manually:")
        print("   pip install sentence-transformers")
    
    print("\n💡 Recommendations:")
    print("- For beginners: Use SentenceTransformers (option 1)")
    print("- For customization: Use Transformers (option 2)")
    print("- For best quality: Use OpenAI (option 3) with API key")

if __name__ == "__main__":
    main()
