import pickle
import os
import numpy as np
from typing import List, Dict, Any
from functools import lru_cache
import threading
import time
import faiss
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger(__name__)

class FastVectorStore:
    """Optimized vector store for faster similarity search using FAISS"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self._embeddings = None
        self._documents = None
        self._metadata = None
        self._loaded = False
        self._lock = threading.Lock()
        self._cache = {}
        self._index = None  # FAISS index
        self._embedding_model = None
        self._embedding_model_name = embedding_model_name
        
    def _lazy_load(self):
        """Load embeddings, build FAISS index, and initialize embedding model only when first needed"""
        if self._loaded:
            return
            
        with self._lock:
            if self._loaded:  # Double-check locking
                return
                
            start_time = time.time()
            logger.info("[FastVectorStore] Loading embeddings...")
            
            try:
                # Try to load Bitext embeddings first
                if os.path.exists("data/bitext_embeddings.pkl"):
                    with open("data/bitext_embeddings.pkl", "rb") as f:
                        data = pickle.load(f)
                        self._embeddings = data.get("embeddings", np.array([]))
                        self._documents = data.get("documents", [])
                        self._metadata = data.get("metadata", [])
                        logger.info(f"[FastVectorStore] Loaded {len(self._documents)} Bitext documents")
                        
                # Try to load product embeddings as fallback
                elif os.path.exists("data/product_embeddings.pkl"):
                    with open("data/product_embeddings.pkl", "rb") as f:
                        data = pickle.load(f)
                        self._embeddings = data.get("embeddings", np.array([]))
                        self._documents = data.get("documents", [])
                        self._metadata = data.get("metadata", [])
                        logger.info(f"[FastVectorStore] Loaded {len(self._documents)} product documents")
                        
                else:
                    # Create empty arrays if no embeddings found
                    self._embeddings = np.array([])
                    self._documents = []
                    self._metadata = []
                    logger.warning("[FastVectorStore] No embeddings found, using empty store")
                
                # Build FAISS index if embeddings are available
                if self._embeddings.size > 0:
                    if not isinstance(self._embeddings, np.ndarray) or self._embeddings.dtype != np.float32:
                        self._embeddings = np.array(self._embeddings, dtype=np.float32)
                    
                    # Ensure embeddings are 2D
                    if self._embeddings.ndim == 1:
                        self._embeddings = self._embeddings.reshape(-1, 1)
                    
                    # Create FAISS index (using IndexFlatL2 for simplicity and exact search)
                    dimension = self._embeddings.shape[1]
                    self._index = faiss.IndexFlatL2(dimension)
                    self._index.add(self._embeddings)  # Add vectors to the index
                    logger.info(f"[FastVectorStore] Built FAISS index with {self._index.ntotal} vectors")
                
                # Initialize embedding model
                logger.info(f"[FastVectorStore] Loading embedding model {self._embedding_model_name}...")
                try:
                    self._embedding_model = SentenceTransformer(self._embedding_model_name)
                    logger.info(f"[FastVectorStore] Embedding model {self._embedding_model_name} loaded")
                except Exception as e:
                    logger.error(f"[FastVectorStore] Error loading embedding model: {str(e)}")
                    raise RuntimeError(f"Failed to load embedding model: {str(e)}")
                
                self._loaded = True
                load_time = time.time() - start_time
                logger.info(f"[FastVectorStore] Embeddings and model loaded in {load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"[FastVectorStore] Error loading embeddings: {str(e)}")
                # Initialize with empty arrays on error
                self._embeddings = np.array([])
                self._documents = []
                self._metadata = []
                self._loaded = True
    
    @lru_cache(maxsize=100)
    def similarity_search(self, query: str, k: int = 3) -> List:
        """Fast similarity search using FAISS with caching, converting query string to embedding"""
        cache_key = f"{query}_{k}"
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Lazy load embeddings, FAISS index, and embedding model
        self._lazy_load()
        
        if len(self._documents) == 0 or self._index is None or self._embedding_model is None:
            return []
        
        try:
            # Convert query string to embedding
            start_time = time.time()
            query_embedding = self._embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            embed_time = time.time() - start_time
            logger.debug(f"[FastVectorStore] Query embedding generated in {embed_time:.2f}s")
            
            # Ensure query_embedding is float32 and 2D
            if not isinstance(query_embedding, np.ndarray) or query_embedding.dtype != np.float32:
                query_embedding = np.array(query_embedding, dtype=np.float32)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Perform FAISS search
            distances, indices = self._index.search(query_embedding, k)
            
            matches = []
            for idx in indices[0]:
                if idx < len(self._documents):  # Ensure index is valid
                    metadata = self._metadata[idx] if idx < len(self._metadata) else {}
                    matches.append(type('Document', (), {
                        'page_content': str(self._documents[idx]),
                        'metadata': metadata
                    })())
            
            # Cache the result
            self._cache[cache_key] = matches
            
            # Limit cache size
            if len(self._cache) > 100:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            return matches
            
        except Exception as e:
            logger.error(f"[FastVectorStore] Search error: {str(e)}")
            return []

# Create global instance
fast_vector_store = FastVectorStore(embedding_model_name="all-MiniLM-L6-v2")

# Backward compatibility
vector_store = fast_vector_store