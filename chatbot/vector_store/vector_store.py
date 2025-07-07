from faiss import IndexFlatL2
import numpy as np
import pickle

class Document:
    """Simple document class to mimic LangChain's Document structure"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class VectorStore:
    def __init__(self):
        self.index = IndexFlatL2(768)  # Adjust to DeepSeek's embedding dimension
        self.documents = []
        self.metadata = []
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
            self.metadata = [{"seller_id": 1, "product_id": 1, "intent": "product_info", "response": "This is a sample product."}]
            self.index.add(np.zeros((1, 768), dtype=np.float32))

    def search(self, query_embedding, seller_id, k=3):
        distances, indices = self.index.search(query_embedding, k)
        filtered_results = [
            self.documents[i] for i in indices[0]
            if self.metadata[i]["seller_id"] == int(seller_id)
         ]
        return filtered_results[:k]
    
    def similarity_search(self, query, k=3):
        """
        LangChain-compatible similarity search method
        Returns Document objects with page_content and metadata
        """
        try:
            # For now, return mock results since we don't have actual embeddings
            # In a real implementation, you would convert query to embeddings first
            results = []
            for i in range(min(k, len(self.documents))):
                doc = Document(
                    page_content=self.documents[i],
                    metadata=self.metadata[i] if i < len(self.metadata) else {}
                )
                results.append(doc)
            return results
        except Exception as e:
            # Return empty results if there's an error
            return []

vector_store = VectorStore()