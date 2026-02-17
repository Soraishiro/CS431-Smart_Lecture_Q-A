"""BM25 sparse retrieval implementation using rank_bm25 library.

Enhanced to store full document metadata for integration with QueryEngine.
Uses simple regex tokenization for Vietnamese text (no heavy NLP dependencies).
"""

import logging
import re
from typing import Dict, List, Tuple, Any, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def simple_vietnamese_tokenize(text: str) -> List[str]:
    """
    Simple Vietnamese tokenizer using regex patterns.

    Fast alternative to underthesea/VnCoreNLP that avoids heavy dependencies.
    Splits on whitespace and punctuation while preserving Vietnamese characters.

    Args:
        text: Input Vietnamese text

    Returns:
        List of tokens
    """
    # Lowercase and normalize
    text = text.lower().strip()

    # Split on whitespace and common punctuation, keep Vietnamese chars
    # Pattern: keep Vietnamese letters (a-z, à-ỹ) and numbers
    tokens = re.findall(
        r"[a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ0-9]+",
        text,
    )

    # Filter out single characters and very short tokens
    tokens = [t for t in tokens if len(t) > 1]

    return tokens


class BM25Index:
    """BM25 sparse retrieval index for Vietnamese text with simple tokenization.

    This class provides BM25 ranking for Vietnamese text documents using
    fast regex-based tokenization (no heavy NLP dependencies).

    Enhanced to store full document metadata (text, video_id, timestamps, etc.)
    for seamless integration with QueryEngine reranking and generation.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        vncorenlp_path: str = None,
        use_vncorenlp: bool = False,
        use_underthesea: bool = False,
    ):
        """Initialize BM25Index with simple Vietnamese tokenizer.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            vncorenlp_path: DEPRECATED - no longer supported
            use_vncorenlp: DEPRECATED - no longer supported
            use_underthesea: DEPRECATED - no longer supported

        Note:
            All tokenization now uses simple_vietnamese_tokenize() for fast startup.
            Heavy NLP tokenizers (VnCoreNLP, underthesea) have been removed.
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = []
        self.corpus = []
        self.metadata_store = {}  # Store full document metadata

        logger.info(f"Initialized BM25Index with simple tokenizer (k1={k1}, b={b})")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text using simple regex tokenization.

        Args:
            text: Input Vietnamese text to tokenize

        Returns:
            List of lowercase word tokens
        """
        return simple_vietnamese_tokenize(text)

    def index_documents(self, docs: List[Dict]) -> None:
        """Index a list of documents for BM25 search.

        Args:
            docs: List of document dictionaries with required fields:
                - chunk_id (str): Unique document identifier
                - text (str): Document text content
                Optional fields (stored in metadata):
                - video_id (str): Video identifier
                - start_time (float): Start timestamp
                - end_time (float): End timestamp
                - enhanced (str): Enhanced/cleaned text
                - Any other metadata fields

        Raises:
            ValueError: If docs is empty or missing required fields
        """
        try:
            if not docs:
                raise ValueError("Cannot index empty document list")

            # Extract doc_ids, texts, and metadata
            self.doc_ids = []
            texts = []
            self.metadata_store = {}

            for doc in docs:
                if "chunk_id" not in doc or "text" not in doc:
                    raise ValueError(
                        "Each document must have 'chunk_id' and 'text' fields"
                    )

                chunk_id = doc["chunk_id"]
                self.doc_ids.append(chunk_id)
                texts.append(doc["text"])

                # Store full metadata for this document
                self.metadata_store[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": doc["text"],
                    "video_id": doc.get("video_id", ""),
                    "start_time": doc.get("start_time", 0.0),
                    "end_time": doc.get("end_time", 0.0),
                    "enhanced": doc.get("enhanced", ""),
                    # Store any additional metadata fields
                    **{
                        k: v
                        for k, v in doc.items()
                        if k
                        not in [
                            "chunk_id",
                            "text",
                            "video_id",
                            "start_time",
                            "end_time",
                            "enhanced",
                        ]
                    },
                }

            # Tokenize all documents
            self.corpus = [self._tokenize(text) for text in texts]

            # Create BM25 index
            self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)

            logger.info(f"Indexed {len(self.doc_ids)} documents with full metadata")

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise

    def add_document(
        self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a single document to the index incrementally.

        Args:
            doc_id: Unique identifier for the document
            text: Document text content
            metadata: Optional metadata dict with fields like video_id, timestamps, etc.

        Raises:
            ValueError: If doc_id or text is empty
        """
        try:
            if not doc_id or not text:
                raise ValueError("doc_id and text must be non-empty")

            # Tokenize the new document
            tokens = self._tokenize(text)

            # Add to corpus and doc_ids
            self.doc_ids.append(doc_id)
            self.corpus.append(tokens)

            # Store metadata
            if metadata is None:
                metadata = {}

            self.metadata_store[doc_id] = {
                "chunk_id": doc_id,
                "text": text,
                "video_id": metadata.get("video_id", ""),
                "start_time": metadata.get("start_time", 0.0),
                "end_time": metadata.get("end_time", 0.0),
                "enhanced": metadata.get("enhanced", ""),
                **{
                    k: v
                    for k, v in metadata.items()
                    if k
                    not in [
                        "chunk_id",
                        "text",
                        "video_id",
                        "start_time",
                        "end_time",
                        "enhanced",
                    ]
                },
            }

            # Rebuild BM25 index with updated corpus
            self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)

            logger.debug(f"Added document {doc_id} to index with metadata")

        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            raise

    def search(
        self, query: str, top_k: int = 10, return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for documents matching the query using BM25 scoring.

        Args:
            query: Search query text
            top_k: Number of top results to return (default: 10)
            return_metadata: If True, return full metadata; if False, return (doc_id, score) tuples

        Returns:
            List of dicts with 'id', 'score', and 'metadata' keys (if return_metadata=True)
            OR List of tuples (doc_id, score) (if return_metadata=False)

            Metadata includes:
                - chunk_id: Document identifier
                - text: Original document text
                - video_id: Video identifier
                - start_time: Start timestamp
                - end_time: End timestamp
                - enhanced: Enhanced/cleaned text
                - Any additional fields from indexing

        Raises:
            RuntimeError: If index has not been initialized
        """
        try:
            if self.bm25 is None:
                raise RuntimeError(
                    "Index not initialized. Call index_documents() first."
                )

            if not query:
                logger.warning("Empty query provided, returning empty results")
                return []

            # Tokenize query
            query_tokens = self._tokenize(query)

            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k results
            top_indices = scores.argsort()[-top_k:][::-1]

            # Filter positive scores
            valid_results = [
                (idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0
            ]

            # Return format based on return_metadata flag
            if return_metadata:
                results = []
                for idx, score in valid_results:
                    doc_id = self.doc_ids[idx]
                    metadata = self.metadata_store.get(
                        doc_id,
                        {
                            "chunk_id": doc_id,
                            "text": "",
                            "video_id": "",
                            "start_time": 0.0,
                            "end_time": 0.0,
                            "enhanced": "",
                        },
                    )

                    results.append({"id": doc_id, "score": score, "metadata": metadata})

                logger.debug(
                    f"BM25 search returned {len(results)} results with metadata "
                    f"for query: {query[:50]}..."
                )
                return results
            else:
                # Legacy format: list of (doc_id, score) tuples
                results = [(self.doc_ids[idx], score) for idx, score in valid_results]
                logger.debug(
                    f"BM25 search returned {len(results)} results (tuples) "
                    f"for query: {query[:50]}..."
                )
                return results

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document.

        Args:
            doc_id: Document identifier

        Returns:
            Metadata dict or None if document not found
        """
        return self.metadata_store.get(doc_id)

    def close(self) -> None:
        """Close resources (no-op with simple tokenizer)."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
