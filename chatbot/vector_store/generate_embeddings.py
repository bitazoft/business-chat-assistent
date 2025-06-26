import os
import requests
import pickle
import numpy as np
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models.schemas import Product
from db.database import Base, engine

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# Database session
SessionLocal = sessionmaker(bind=engine)

def generate_deepseek_embeddings(texts: list, api_key: str, api_base: str) -> np.ndarray:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    embeddings = []
    for text in texts:
        try:
            payload = {"input": text, "model": "deepseek-embedding"}  # Replace with DeepSeek's actual embedding model
            response = requests.post(f"{api_base}/embeddings", json=payload, headers=headers)
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for text '{text}': {str(e)}")
            embeddings.append(np.zeros(768))  # Placeholder for failed embeddings; adjust dimension as needed
    return np.array(embeddings, dtype=np.float32)

def main():
    # Fetch products from database
    db = SessionLocal()
    try:
        products = db.query(Product).all()
        if not products:
            print("No products found in database")
            return

        # Prepare texts and metadata
        texts = [f"{p.name}: {p.description or ''}" for p in products]
        metadata = [{"seller_id": p.seller_id, "product_id": p.id} for p in products]

        # Generate embeddings
        embeddings = generate_deepseek_embeddings(texts, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE)

        # Save to file
        os.makedirs("data", exist_ok=True)
        with open("data/product_embeddings.pkl", "wb") as f:
            pickle.dump({"embeddings": embeddings, "documents": texts, "metadata": metadata}, f)
        print("Embeddings generated and saved successfully")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    main()