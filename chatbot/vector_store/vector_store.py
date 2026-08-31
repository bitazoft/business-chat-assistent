"""
FAISS-backed similarity search over the example corpus.

Three things changed from the previous version:

1. faiss and sentence_transformers are imported inside _lazy_load, not at module
   top level. They pull in torch - several hundred MB and a few seconds of import
   time - and this module is imported by the agent on every start even when
   RAG_ENABLED is false. The app now boots without them installed.

2. similarity_search no longer carries @lru_cache. Decorating a method caches on
   (self, ...), which pins the instance - and every embedding array it holds - for
   the life of the process, and it sat on top of a second hand-rolled dict cache
   that did the same job with different keys. One TTL cache now.

3. Stored vectors are L2-normalised to match the query encoding. Queries were
   encoded with normalize_embeddings=True while the index held raw vectors, so
   the distances being compared against the threshold were between vectors of
   different magnitudes and the threshold meant nothing consistent.
"""
import os
import pickle
import threading
import time
from typing import Any, Dict, List

import numpy as np

from utils.cache import TTLCache
from utils.logger import get_logger

logger = get_logger(__name__)


class Document:
    """A search hit. A real class, not a type() built per result."""

    __slots__ = ("page_content", "metadata", "similarity_score")

    def __init__(self, page_content: str, metadata: Dict[str, Any], similarity_score: float):
        self.page_content = page_content
        self.metadata = metadata
        self.similarity_score = similarity_score

    def __repr__(self) -> str:
        return f"Document(score={self.similarity_score:.4f}, content={self.page_content[:40]!r})"


class FastVectorStore:
    """Optimized vector store for faster similarity search using FAISS"""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self._embeddings = None
        self._documents: List[Any] = []
        self._metadata: List[Dict[str, Any]] = []
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()
        self._index = None  # FAISS index
        self._embedding_model = None
        self._embedding_model_name = embedding_model_name
        self._cache = TTLCache(maxsize=500, ttl=1800, name="vector_search")

    def _lazy_load(self) -> None:
        """Load embeddings, build the FAISS index, and init the embedding model."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:  # Double-check locking
                return

            start_time = time.time()
            logger.info("[FastVectorStore] Loading embeddings...")

            try:
                # Imported here, not at module scope: these pull in torch.
                import faiss
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                logger.warning(
                    "[FastVectorStore] Vector search unavailable (%s). "
                    "Install sentence-transformers and faiss-cpu to enable RAG.",
                    e,
                )
                self._mark_empty()
                return

            try:
                self._load_corpus()

                if self._embeddings is not None and self._embeddings.size > 0:
                    embeddings = np.asarray(self._embeddings, dtype=np.float32)
                    if embeddings.ndim == 1:
                        embeddings = embeddings.reshape(1, -1)

                    # Match the query-side normalisation so L2 distance is
                    # comparable across rows and the threshold means something.
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    embeddings = embeddings / norms

                    self._embeddings = embeddings
                    self._index = faiss.IndexFlatL2(embeddings.shape[1])
                    self._index.add(embeddings)
                    logger.info(
                        "[FastVectorStore] Built FAISS index with %d vectors",
                        self._index.ntotal,
                    )

                logger.info(
                    "[FastVectorStore] Loading embedding model %s...",
                    self._embedding_model_name,
                )
                self._embedding_model = SentenceTransformer(self._embedding_model_name)

                self._loaded = True
                logger.info(
                    "[FastVectorStore] Embeddings and model loaded in %.2fs",
                    time.time() - start_time,
                )

            except Exception as e:
                logger.error("[FastVectorStore] Error loading embeddings: %s", e)
                self._mark_empty()

    def _load_corpus(self) -> None:
        """Read the pickled corpus, preferring Bitext over product embeddings."""
        for path, label in (
            ("data/bitext_embeddings.pkl", "Bitext"),
            ("data/product_embeddings.pkl", "product"),
        ):
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._embeddings = data.get("embeddings", np.array([]))
            self._documents = data.get("documents", [])
            self._metadata = data.get("metadata", [])
            logger.info(
                "[FastVectorStore] Loaded %d %s documents", len(self._documents), label
            )
            return

        self._embeddings = np.array([])
        self._documents = []
        self._metadata = []
        logger.warning("[FastVectorStore] No embeddings found, using empty store")

    def _mark_empty(self) -> None:
        """Settle into a working-but-empty state. Caller must hold the lock."""
        self._embeddings = np.array([])
        self._documents = []
        self._metadata = []
        self._index = None
        self._embedding_model = None
        self._loaded = True
        self._load_failed = True

    @property
    def is_available(self) -> bool:
        """True when a search could actually return something."""
        return bool(self._loaded and self._index is not None and self._embedding_model)

    def similarity_search(self, query: str, k: int = 3, threshold: float = 3) -> List[Document]:
        """Fast similarity search using FAISS with caching.

        Args:
            query: Query string.
            k: Number of top matches to return.
            threshold: Maximum L2 distance for a match to count.

        Returns:
            Matching documents, closest first. Empty list if the store is empty
            or unavailable - a search problem should degrade the prompt, not fail
            the customer's message.
        """
        if not query:
            return []

        cache_key = f"{k}|{threshold}|{query.strip().lower()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        self._lazy_load()

        if not self.is_available or not self._documents:
            return []

        try:
            start_time = time.time()
            query_embedding = self._embedding_model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )
            logger.debug(
                "[FastVectorStore] Query embedding generated in %.3fs",
                time.time() - start_time,
            )

            query_embedding = np.asarray(query_embedding, dtype=np.float32)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            distances, indices = self._index.search(
                query_embedding, min(k, len(self._documents))
            )

            matches: List[Document] = []
            for distance, idx in zip(distances[0], indices[0]):
                # FAISS returns -1 when it has fewer neighbours than requested.
                if idx < 0 or idx >= len(self._documents) or distance > threshold:
                    continue
                matches.append(
                    Document(
                        page_content=str(self._documents[idx]),
                        metadata=self._metadata[idx] if idx < len(self._metadata) else {},
                        similarity_score=float(distance),
                    )
                )

            self._cache.set(cache_key, matches)
            return matches

        except Exception as e:
            logger.error("[FastVectorStore] Search error: %s", e)
            return []

    def stats(self) -> Dict[str, Any]:
        """Exposed at /metrics so it's obvious when RAG is silently doing nothing."""
        return {
            "loaded": self._loaded,
            "load_failed": self._load_failed,
            "available": self.is_available,
            "documents": len(self._documents),
            "model": self._embedding_model_name,
            "cache": self._cache.stats(),
        }


# Create global instance
fast_vector_store = FastVectorStore(embedding_model_name="all-MiniLM-L6-v2")

# Backward compatibility
vector_store = fast_vector_store
