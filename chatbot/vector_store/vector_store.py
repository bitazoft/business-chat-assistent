from faiss import IndexFlatL2
import numpy as np
import pickle

class VectorStore:
    def __init__(self):
        self.index = IndexFlatL2(768)  # Adjust to DeepSeek's embedding dimension
        self.documents = []
        self.load_vectors()

    def load_vectors(self):
        try:
            with open("data/product_embeddings.pkl", "rb") as f:
                data = pickle.load(f)
                self.index.add(data["embeddings"])
                self.documents = data["documents"]
                self.metadata = data["metadata"]
        except:
            self.documents = ["Sample product description"]
            self.metadata = [{"seller_id": 1, "product_id": 1}]
            self.index.add(np.zeros((1, 768), dtype=np.float32))

    def search(self, query_embedding, seller_id, k=3):
        distances, indices = self.index.search(query_embedding, k)
        filtered_results = [
            self.documents[i] for i in indices[0]
            if self.metadata[i]["seller_id"] == int(seller_id)
         ]
        return filtered_results[:k]

vector_store = VectorStore()