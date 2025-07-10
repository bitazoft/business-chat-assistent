import sys
import os
import unittest
from vector_store.vector_store import FastVectorStore

class TestFastVectorStore(unittest.TestCase):

    def setUp(self):
        self.vector_store = FastVectorStore()

    def test_similarity_search_empty_store(self):
        """Test similarity search when the store is empty."""
        query = "test query"
        results = self.vector_store.similarity_search(query, k=3)
        self.assertEqual(results, [], "Expected empty result for empty store")

    def test_similarity_search_with_mock_data(self):
        """Test similarity search with mock data."""
        # Mock data setup
        self.vector_store._documents = ["doc1", "doc2", "doc3"]
        self.vector_store._metadata = [{"id": 1}, {"id": 2}, {"id": 3}]
        self.vector_store._embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        self.vector_store._index = None  # Reset index
        self.vector_store._lazy_load()  # Rebuild index

        query = "I need to buy product 1"
        results = self.vector_store.similarity_search(query, k=2)
        self.assertTrue(len(results) <= 2, "Expected at most 2 results")

    def test_similarity_search_with_real_dataset(self):
        """Test similarity search with a real dataset and multiple queries."""
        # Load real dataset
        self.vector_store._documents = ["Product 1 description", "Product 2 description", "Product 3 description"]
        self.vector_store._metadata = [{"id": 1}, {"id": 2}, {"id": 3}]
        self.vector_store._embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        self.vector_store._index = None  # Reset index
        self.vector_store._lazy_load()  # Rebuild index

        # Multiple queries
        queries = ["I need product 1", "Tell me about product 2", "Looking for product 3"]
        for query in queries:
            results = self.vector_store.similarity_search(query, k=2)
            print(f"Query: {query}\nResults: {results}\n")

if __name__ == "__main__":
    unittest.main()
