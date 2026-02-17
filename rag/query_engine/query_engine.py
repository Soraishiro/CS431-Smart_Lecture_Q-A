"""
Query engine orchestrating the complete RAG retrieval pipeline.

TODO (Refactoring):
- Remove file-based caching (use Redis instead)
- Simplify to single embedding model (ME5)
- Extract retriever, reranker, generator into separate service classes
- Add proper error handling and retry logic
- Implement request/response models with Pydantic
"""

import logging
import hashlib
import json
import pickle
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag.db.interface import VectorDBInterface
from rag.bm25.bm25_index import BM25Index
from rag.query_engine.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


# ==============================================================================
# FUTURE FEATURE: Ensemble Retrieval (3 Models)
# ==============================================================================
# TODO: Integrate ensemble retrieval as an alternative to single-model BGE
#
# Current: Single BGE model (1024d) + BM25
# Future: 3-model ensemble (BGE + Vietnamese + ME5) + BM25
#
# To enable:
# 1. Uncomment EnsembleRetriever class below
# 2. Add multi-collection support to VectorDBInterface.search()
# 3. Update QueryEngine.__init__ to accept ensemble_mode flag
# 4. Modify _retrieve_candidates to use ensemble when enabled
# 5. Test with all 3 collections indexed (cs431_bge, cs431_vietnamese, cs431_me5)
#
# Benefits:
# - Better multilingual retrieval (Vietnamese-specific model)
# - Instruction-following queries (ME5 model)
# - Ensemble robustness via RRF fusion
#
# class EnsembleRetriever:
#     """FUTURE: Ensemble retriever combining 3 embedding models + BM25."""
#
#     def __init__(self, vector_db, bm25_index, device='cuda'):
#         from sentence_transformers import SentenceTransformer
#         self.db = vector_db
#         self.bm25 = bm25_index
#
#         # Load 3 embedding models
#         self.model_bge = SentenceTransformer('BAAI/bge-m3', device=device)
#         self.model_viet = SentenceTransformer('dangvantuan/vietnamese-document-embedding', device=device)
#         self.model_me5 = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=device)
#
#     def retrieve(self, query: str, top_k: int = 100) -> Dict[str, List[Dict]]:
#         """Retrieve using 3-model ensemble + BM25."""
#         # Encode query with all 3 models
#         query_bge = self.model_bge.encode(query, normalize_embeddings=True)
#         query_viet = self.model_viet.encode(query, normalize_embeddings=True)
#         query_me5 = self.model_me5.encode(query, normalize_embeddings=True)
#
#         # Query each collection (requires multi-collection support in VectorDB)
#         # results_bge = self.db.search(query_bge.tolist(), top_k=top_k, collection_name='cs431_bge')
#         # results_viet = self.db.search(query_viet.tolist(), top_k=top_k, collection_name='cs431_vietnamese')
#         # results_me5 = self.db.search(query_me5.tolist(), top_k=top_k, collection_name='cs431_me5')
#
#         # Merge semantic results with RRF
#         # semantic_results = self._rrf_fusion([results_bge, results_viet, results_me5])
#
#         # BM25 search
#         # bm25_results = self.bm25.search(query, top_k=top_k)
#
#         # return {'vector': semantic_results, 'bm25': bm25_results}
#         pass
#
#     def _rrf_fusion(self, result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
#         """Merge multiple result lists using Reciprocal Rank Fusion."""
#         scores = {}
#         for results in result_lists:
#             for rank, result in enumerate(results, 1):
#                 chunk_id = result['id']
#                 if chunk_id not in scores:
#                     scores[chunk_id] = {'score': 0, 'result': result}
#                 scores[chunk_id]['score'] += 1 / (k + rank)
#
#         merged = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
#         return [item['result'] for item in merged]
# ==============================================================================


class QueryEngine:
    """Orchestrates the complete RAG retrieval and generation pipeline.

    Pipeline flow:
    1. Embed query with BGE
    2. Parallel retrieval: vector search (top 100) + BM25 (top 100)
    3. RRF fusion → top 50
    4. Rerank → top 20
    5. Generate answer with Gemini 2.5
    """

    def __init__(
        self,
        vector_db: VectorDBInterface,
        bm25_index: BM25Index,
        reranker: Any,  # CrossEncoderReranker or KaggleReranker
        embedder: Any,  # FastEmbedder, MultiModelEmbedder, or KaggleEmbedder
        gemini_client: Any,  # GeminiClient, KaggleGeminiClient, or KaggleLLMClient
        cache_dir: str = "./cache",
        enable_cache: bool = True,
    ):
        """
        Initialize the query engine with all required components.

        Args:
            vector_db: Vector database instance (Pinecone or Milvus)
            bm25_index: BM25 sparse retrieval index
            reranker: Cross-encoder reranker (local or Kaggle adapter)
            embedder: Embedding model (local or Kaggle adapter)
            gemini_client: Gemini client for answer generation (local or Kaggle adapter)
            cache_dir: Directory for caching results (default: ./cache)
            enable_cache: Enable caching for faster repeated queries

        Raises:
            ValueError: If any component is None or invalid

        Note:
            This engine supports both local and Kaggle-hosted inference:
            - Local: CrossEncoderReranker, FastEmbedder/MultiModelEmbedder, GeminiClient
            - Kaggle: KaggleReranker, KaggleEmbedder, KaggleGeminiClient

            Components must implement these interfaces:
            - embedder: embed_query(query: str) -> List[float] or np.ndarray
            - reranker: rerank(query: str, documents: List[Dict], top_k: int) -> List[Dict]
            - gemini_client: generate_answer(query: str, context_chunks: List[str], temperature: float) -> str
        """
        # Validate components
        if vector_db is None:
            raise ValueError("vector_db cannot be None")
        if bm25_index is None:
            raise ValueError("bm25_index cannot be None")
        if reranker is None:
            raise ValueError("reranker cannot be None")
        if embedder is None:
            raise ValueError("embedder cannot be None")
        if gemini_client is None:
            raise ValueError("gemini_client cannot be None")

        self.vector_db = vector_db
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.embedder = embedder
        self.gemini_client = gemini_client

        # Cache setup
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir)
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache enabled: {self.cache_dir}")

        logger.info("QueryEngine initialized successfully with all components")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text using MD5 hash."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_from_cache(self, cache_key: str, cache_type: str) -> Optional[Any]:
        """Load data from cache file."""
        if not self.enable_cache:
            return None

        cache_file = self.cache_dir / f"{cache_type}_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"✓ Cache hit: {cache_type}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, cache_type: str, data: Any):
        """Save data to cache file."""
        if not self.enable_cache:
            return

        cache_file = self.cache_dir / f"{cache_type}_{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved to cache: {cache_type}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _embed_query(
        self, query: str, model: str = "vietnamese"
    ) -> Union[List[float], Dict[str, List[float]]]:
        """
        Embed the query text using embedding model(s) (with caching).

        Args:
            query: User query text
            model: Model to use ("bge", "vietnamese", "me5", or "all")

        Returns:
            Single embedding vector if model is specific, or dict of embeddings if model="all"
            - "vietnamese": 768-dim
            - "bge": 1024-dim
            - "me5": 1024-dim
            - "all": {"vietnamese": [...], "bge": [...], "me5": [...]}

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(f"{query}_{model}")
            cached_embedding = self._load_from_cache(cache_key, "embedding")
            if cached_embedding is not None:
                return cached_embedding

            logger.info(f"Embedding query with model={model}: {query[:10]}...")

            # Use embedder interface (supports local and Kaggle)
            if hasattr(self.embedder, "embed_query"):
                # Single model (FastEmbedder, KaggleEmbedder)
                embedding_result = self.embedder.embed_query(query, model=model)

                # Handle multi-model response
                if isinstance(embedding_result, dict):
                    # Multi-model (model="all")
                    embedding = {
                        k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in embedding_result.items()
                    }
                    logger.info(f"Query embedded with all models")
                else:
                    # Single model
                    if hasattr(embedding_result, "tolist"):
                        embedding = embedding_result.tolist()
                    else:
                        embedding = embedding_result
                    logger.info(f"Query embedded successfully (dim={len(embedding)})")
            elif hasattr(self.embedder, "embed_single"):
                # MultiModelEmbedder (local multi-model)
                result = self.embedder.embed_single(query)

                if model == "all":
                    # Return all three embeddings
                    embedding = {
                        "vietnamese": result.vietnamese_embedding.tolist(),
                        "bge": result.bge_embedding.tolist(),
                        "me5": result.me5_embedding.tolist(),
                    }
                    logger.info(f"Query embedded with all 3 models")
                elif model == "bge":
                    embedding = result.bge_embedding.tolist()
                    logger.info(f"Query embedded with BGE (dim={len(embedding)})")
                elif model == "me5":
                    embedding = result.me5_embedding.tolist()
                    logger.info(f"Query embedded with ME5 (dim={len(embedding)})")
                else:  # vietnamese (default)
                    embedding = result.vietnamese_embedding.tolist()
                    logger.info(
                        f"Query embedded with Vietnamese (dim={len(embedding)})"
                    )
            else:
                raise RuntimeError(
                    "Embedder does not have embed_query or embed_single method"
                )

            # Save to cache
            self._save_to_cache(cache_key, "embedding", embedding)

            return embedding

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Query embedding failed: {e}") from e

    async def _retrieve_candidates_async(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 20,
        collection_suffix: str = "",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve candidates from both vector database and BM25 index in parallel (async).

        Args:
            query_vector: Query embedding vector
            query_text: Original query text for BM25
            top_k: Number of results to retrieve from each source (default: 20)
            collection_suffix: Model name suffix for collection (e.g., "bge", "vietnamese", "me5")

        Returns:
            Dictionary with 'vector' and 'bm25' keys containing result lists
        """
        logger.info(f"Starting async parallel retrieval (top_k={top_k})")

        async def retrieve_vector():
            """Retrieve from vector database."""
            try:
                logger.info(f"Retrieving from vector database (model={collection_suffix})...")
                
                # Determine index and namespace based on model
                index_name = None
                namespace = ""
                
                if collection_suffix == "vietnamese":
                    index_name = "cs431-embeddings-768"
                    namespace = "vietnamese"
                elif collection_suffix == "me5":
                    index_name = "cs431-embeddings"
                    namespace = "me5"
                elif collection_suffix == "bge":
                    index_name = "cs431-embeddings"
                    namespace = "bge"
                
                # Run blocking I/O in thread pool
                loop = asyncio.get_event_loop()
                # Use lambda to pass kwargs to search
                results = await loop.run_in_executor(
                    None, 
                    lambda: self.vector_db.search(
                        query_vector, 
                        top_k, 
                        namespace=namespace, 
                        index_name=index_name
                    )
                )
                logger.info(f"Vector search returned {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

        async def retrieve_bm25():
            """Retrieve from BM25 index."""
            try:
                logger.info("Retrieving from BM25 index...")
                # Run blocking I/O in thread pool
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, self.bm25_index.search, query_text, top_k, True
                )
                logger.info(
                    f"BM25 search returned {len(results)} results with metadata"
                )
                return results
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")
                return []

        # Execute both retrievals in parallel using asyncio.gather
        vector_results, bm25_results = await asyncio.gather(
            retrieve_vector(), retrieve_bm25(), return_exceptions=False
        )

        logger.info(
            f"Async parallel retrieval complete: "
            f"vector={len(vector_results)}, bm25={len(bm25_results)}"
        )

        return {"vector": vector_results, "bm25": bm25_results}

    def _retrieve_candidates(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 20,
        collection_suffix: str = "",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Synchronous wrapper for async retrieval.

        Args:
            query_vector: Query embedding vector
            query_text: Query text for BM25
            top_k: Number of results to retrieve
            collection_suffix: Model name suffix for collection (e.g., "bge", "vietnamese", "me5")
        """
        # Check cache first
        cache_key = self._get_cache_key(
            json.dumps(
                {
                    "vector": query_vector[:10],  # Use first 10 dims for key
                    "text": query_text,
                    "top_k": top_k,
                    "collection": collection_suffix,
                }
            )
        )
        cached_results = self._load_from_cache(cache_key, "retrieval")
        if cached_results is not None:
            return cached_results

        # Run async retrieval
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(
            self._retrieve_candidates_async(
                query_vector, query_text, top_k, collection_suffix
            )
        )

        # Save to cache
        self._save_to_cache(cache_key, "retrieval", results)

        return results

    def _fusion_rrf(
        self,
        result_lists: List[List[Dict[str, Any]]],
        k: int = 60,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple ranked result lists using Reciprocal Rank Fusion.

        RRF formula: score(d) = Σ[1/(k + rank)] across all result lists
        where rank starts at 1 for each list.

        Args:
            result_lists: List of result lists (e.g., [bge_results, viet_results, me5_results, bm25_results])
            k: RRF constant (default: 60)
            top_k: Number of top fused results to return (default: 50)

        Returns:
            List of top-k fused candidates sorted by RRF score descending
        """
        logger.info(
            f"Applying Reciprocal Rank Fusion across {len(result_lists)} result lists (k={k})"
        )
        # Dictionary to accumulate RRF scores by chunk_id
        rrf_scores = {}
        chunk_data = {}  # Store full chunk data

        # Process each result list
        for list_idx, results in enumerate(result_lists):
            for rank, result in enumerate(results, start=1):
                # IMPORTANT: Check metadata.chunk_id FIRST (the true chunk ID)
                # before falling back to id (which could be Pinecone's internal ID)
                chunk_id = result.get("metadata", {}).get("chunk_id") or result.get("id")
                if not chunk_id:
                    continue

                # RRF score contribution: 1 / (k + rank)
                score = 1.0 / (k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score

                # Store chunk data if not already present (prefer first occurrence)
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = result

        # Create fused results with RRF scores
        fused_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            chunk = chunk_data[chunk_id].copy()
            chunk["rrf_score"] = rrf_score
            chunk["chunk_id"] = chunk_id
            fused_results.append(chunk)

        # Sort by RRF score descending and return top-k
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_fused = fused_results[:top_k]

        logger.info(
            f"RRF fusion complete: {len(rrf_scores)} unique chunks, "
            f"returning top {len(top_fused)}"
        )

        return top_fused

    def process_query(
        self,
        query: str,
        video_id: Optional[str] = None,
        embed_model: str = "vietnamese",
        vector_top_k: int = 100,
        bm25_top_k: int = 100,
        fusion_top_k: int = 50,
        rerank_top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.

        Pipeline steps:
        1. Embed query with selected model(s)
        2. Parallel retrieval: vector (100) + BM25 (100)
        3. RRF fusion across all result lists → top 50
        4. Rerank → top 20
        5. Generate answer with Gemini

        Args:
            query: User query text
            video_id: Optional video ID filter
            embed_model: Embedding model ("bge", "vietnamese", "me5", or "all")

        Returns:
            Dictionary containing:
                - answer: Generated answer with citations
                - contexts: Top reranked chunks used for generation
                - metadata: Pipeline execution metadata

        Raises:
            RuntimeError: If pipeline execution fails critically
        """
        logger.info(
            f"Processing query with embed_model={embed_model}: {query[:100]}..."
        )

        try:
            # Step 1: Embed query
            logger.info("Step 1/5: Embedding query...")
            query_embeddings = self._embed_query(query, model=embed_model)

            # Step 2: Multi-model parallel retrieval
            logger.info("Step 2/5: Multi-model parallel retrieval (vector + BM25)...")

            result_lists = []
            bm25_results = []
            total_vector_count = 0

            # Handle multi-model retrieval
            if isinstance(query_embeddings, dict):
                # Multi-model: retrieve from each model's collection
                for i, (model_name, embedding) in enumerate(query_embeddings.items()):
                    logger.info(f"   Retrieving from {model_name} collection...")
                    model_results = self._retrieve_candidates(
                        query_vector=embedding,
                        query_text=query,
                        top_k=vector_top_k,
                        collection_suffix=model_name,
                    )
                    if model_results.get("vector"):
                        result_lists.append(model_results["vector"])
                        total_vector_count += len(model_results["vector"])
                    
                    # Capture BM25 results from the first model (it's the same index)
                    if i == 0 and model_results.get("bm25"):
                        bm25_results = model_results["bm25"]
            else:
                # Single model: use default or model-specific collection
                retrieval_results = self._retrieve_candidates(
                    query_vector=query_embeddings,
                    query_text=query,
                    top_k=vector_top_k,
                    collection_suffix=embed_model,
                )
                if retrieval_results.get("vector"):
                    result_lists.append(retrieval_results["vector"])
                    total_vector_count = len(retrieval_results["vector"])
                
                # Capture BM25 results
                if retrieval_results.get("bm25"):
                    bm25_results = retrieval_results["bm25"]

            # Add BM25 results to the list for fusion
            if bm25_results:
                result_lists.append(bm25_results)

            # Optional filtering: restrict to chosen video if provided
            if video_id:
                logger.info(f"Filtering results by video_id: {video_id}")
                result_lists = [
                    [
                        r
                        for r in results
                        if r.get("metadata", {}).get("video_id") == video_id
                    ]
                    for results in result_lists
                ]

            # Check if we have any results
            if not any(result_lists):
                logger.warning("No results from retrieval, returning empty answer")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "contexts": [],
                    "metadata": {
                        "query": query,
                        "vector_count": 0,
                        "bm25_count": len(bm25_results) if bm25_results else 0,
                        "fused_count": 0,
                        "reranked_count": 0,
                    },
                }

            # Step 3: RRF fusion across all result lists → top 50
            logger.info(
                "Step 3/5: Applying Reciprocal Rank Fusion across all result lists..."
            )
            fused_results = self._fusion_rrf(
                result_lists=result_lists, k=60, top_k=fusion_top_k
            )

            if not fused_results:
                logger.warning("No results after fusion, returning empty answer")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "contexts": [],
                    "metadata": {
                        "query": query,
                        "embed_model": embed_model,
                        "vector_count": total_vector_count,
                        "bm25_count": len(bm25_results) if bm25_results else 0,
                        "fused_count": 0,
                        "reranked_count": 0,
                    },
                }

            # Step 4: Rerank with cross-encoder
            logger.info("Step 4/5: Reranking with cross-encoder...")

            # Prepare chunks for reranking (need to extract text and metadata)
            chunks_for_reranking = []
            skipped_empty = 0
            for result in fused_results:
                metadata = result.get("metadata", {})
                # IMPORTANT: Prefer metadata.chunk_id (true ID) over result fields
                true_chunk_id = (
                    metadata.get("chunk_id") 
                    or result.get("chunk_id") 
                    or result.get("id")
                )
                
                # Get text - try both 'text' and 'enhanced'
                text = metadata.get("text", "") or metadata.get("enhanced", "")
                
                # Skip chunks with empty text (would fail reranker validation)
                if not text or not text.strip():
                    skipped_empty += 1
                    logger.warning(f"Skipping chunk {true_chunk_id} - empty text field")
                    continue
                
                chunk = {
                    "chunk_id": true_chunk_id,
                    "text": text,
                    "video_id": metadata.get("video_id", ""),
                    "start_time": metadata.get("start_time", 0.0),
                    "end_time": metadata.get("end_time", 0.0),
                }
                chunks_for_reranking.append(chunk)
            
            if skipped_empty > 0:
                logger.warning(f"Skipped {skipped_empty} chunks with empty text")
            
            if not chunks_for_reranking:
                logger.error("No valid chunks to rerank (all had empty text)")
                return {
                    "answer": "I couldn't find any valid information to answer your question.",
                    "contexts": [],
                    "metadata": {
                        "query": query,
                        "error": "No valid chunks for reranking"
                    }
                }

            reranked_chunks = self.reranker.rerank(
                query=query, documents=chunks_for_reranking, top_k=rerank_top_k
            )

            if not reranked_chunks:
                logger.warning("No results after reranking, using fused results")
                reranked_chunks = chunks_for_reranking[:5]

            # Smart context selection: use positive scores or minimum 5
            contexts_for_generation = []
            for chunk in reranked_chunks:
                score = chunk.get("rerank_score", 0.0)
                if score > 0.0:  # Only positive scores
                    contexts_for_generation.append(chunk)

            # Ensure minimum 5 contexts
            if len(contexts_for_generation) < 5 and len(reranked_chunks) >= 5:
                contexts_for_generation = reranked_chunks[:5]
            elif len(contexts_for_generation) == 0:
                contexts_for_generation = reranked_chunks

            logger.info(
                f"Using {len(contexts_for_generation)} contexts for generation "
                f"(positive scores or min 5)"
            )

            # Step 5: Generate answer with Gemini
            logger.info("Step 5/5: Generating answer with Gemini...")

            # Pass full context dicts (not just text) for better citation support
            # Each dict should have: text, start_time, end_time, video_id
            answer = self.gemini_client.generate_answer(
                query=query, context_chunks=contexts_for_generation, temperature=0.3
            )

            logger.info("Query processing complete!")

            return {
                "answer": answer,
                "contexts": contexts_for_generation,  # Return only contexts used for generation
                "metadata": {
                    "query": query,
                    "embed_model": embed_model,
                    "vector_count": total_vector_count,
                    "bm25_count": len(bm25_results) if bm25_results else 0,
                    "fused_count": len(fused_results),
                    "reranked_count": len(reranked_chunks),
                    "generation_context_count": len(contexts_for_generation),  # Track actual count used
                },
            }

        except RuntimeError as e:
            # Re-raise runtime errors (e.g., embedding failures)
            logger.error(f"Pipeline failed with RuntimeError: {e}")
            raise
        except Exception as e:
            # Catch all other exceptions and wrap them
            logger.error(f"Unexpected error in query pipeline: {e}", exc_info=True)
            raise RuntimeError(f"Query processing failed: {e}") from e

    def retrieve_only(
        self,
        query: str,
        video_id: Optional[str] = None,
        top_k: int = 20,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        fusion_top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Retrieve and rerank relevant chunks without generating an answer.

        This method is useful for:
        - Debugging retrieval pipeline
        - Building custom UIs that need raw chunks
        - Pre-fetching contexts for manual review

        Args:
            query: User query text
            video_id: Optional video ID to filter results
            top_k: Number of final reranked results to return (default: 20)
            vector_top_k: Number of vector search results (default: 20)
            bm25_top_k: Number of BM25 results (default: 20)
            fusion_top_k: Number of results after fusion (default: 10)

        Returns:
            Dictionary containing:
                - results: List of reranked chunks with scores
                - metadata: Retrieval pipeline metadata

        Raises:
            RuntimeError: If retrieval fails
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        try:
            # Step 1: Embed query (use Vietnamese by default for retrieve_only)
            logger.info("Step 1/3: Embedding query...")
            query_vector = self._embed_query(query, model="vietnamese")

            # Handle case where multi-model embedding was returned
            if isinstance(query_vector, dict):
                query_vector = query_vector["vietnamese"]

            # Step 2: Parallel retrieval
            logger.info(
                f"Step 2/3: Parallel retrieval (vector={vector_top_k}, bm25={bm25_top_k})..."
            )
            retrieval_results = self._retrieve_candidates(
                query_vector=query_vector,
                query_text=query,
                top_k=max(vector_top_k, bm25_top_k),
            )

            vector_results = retrieval_results["vector"][:vector_top_k]
            bm25_results = retrieval_results["bm25"][:bm25_top_k]

            # Optional filtering by video_id
            if video_id:
                logger.info(f"Filtering results by video_id: {video_id}")
                vector_results = [
                    r
                    for r in vector_results
                    if r.get("metadata", {}).get("video_id") == video_id
                ]
                bm25_results = [
                    r
                    for r in bm25_results
                    if r.get("metadata", {}).get("video_id") == video_id
                ]

            # Check if we have any results
            if not vector_results and not bm25_results:
                logger.warning("No results from retrieval")
                return {
                    "results": [],
                    "metadata": {
                        "query": query,
                        "vector_count": 0,
                        "bm25_count": 0,
                        "fused_count": 0,
                        "reranked_count": 0,
                    },
                }

            # Step 3: RRF fusion
            logger.info(
                f"Step 3/3: Fusion (top_k={fusion_top_k}) and reranking (top_k={top_k})..."
            )
            fused_results = self._fusion_rrf(
                result_lists=[vector_results, bm25_results],
                k=60,
                top_k=fusion_top_k,
            )

            if not fused_results:
                logger.warning("No results after fusion")
                return {
                    "results": [],
                    "metadata": {
                        "query": query,
                        "vector_count": len(vector_results),
                        "bm25_count": len(bm25_results),
                        "fused_count": 0,
                        "reranked_count": 0,
                    },
                }

            # Step 4: Rerank
            chunks_for_reranking = []
            for result in fused_results:
                metadata = result.get("metadata", {})
                # IMPORTANT: Prefer metadata.chunk_id (true ID) over result fields
                true_chunk_id = (
                    metadata.get("chunk_id") 
                    or result.get("chunk_id") 
                    or result.get("id")
                )
                chunk = {
                    "chunk_id": true_chunk_id,
                    "text": metadata.get("text", ""),
                    "video_id": metadata.get("video_id", ""),
                    "start_time": metadata.get("start_time", 0.0),
                    "end_time": metadata.get("end_time", 0.0),
                }
                chunks_for_reranking.append(chunk)

            reranked_chunks = self.reranker.rerank(
                query=query, documents=chunks_for_reranking, top_k=top_k
            )

            logger.info(f"Retrieval complete: {len(reranked_chunks)} results")

            return {
                "results": reranked_chunks,
                "metadata": {
                    "query": query,
                    "vector_count": len(vector_results),
                    "bm25_count": len(bm25_results),
                    "fused_count": len(fused_results),
                    "reranked_count": len(reranked_chunks),
                },
            }

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            raise RuntimeError(f"Retrieval failed: {e}") from e
