import sys
import os
from vector_store.vector_store import FastVectorStore

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Initialize the vector store
vector_store = FastVectorStore()

# # Load real dataset
# vector_store._documents = ["Product 1 description", "Product 2 description", "Product 3 description"]
# vector_store._metadata = [{"id": 1}, {"id": 2}, {"id": 3}]
# vector_store._embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
# vector_store._index = None  # Reset index
# vector_store._lazy_load()  # Rebuild index

# Multiple queries
queries = [ "Tell me about product 2", "What can you do","Can you tarack my order", "My product has defects"]
for query in queries:
    results = vector_store.similarity_search(query, k=2)
    for result in results:
        page_content = result.page_content
        metadata = result.metadata
        print(f"Query: {query}\nPage Content: {page_content}\nMetadata: {metadata}\n")
