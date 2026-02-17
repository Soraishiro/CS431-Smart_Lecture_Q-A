"""
Video Q&A RAG API Service

Provides REST API endpoints for:
- Query processing with hybrid retrieval and answer generation
- Health checks and system status
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pickle
import logging
from typing import Optional, Dict, Any
import time
from pathlib import Path

# Add modules to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# Lazy imports - only import when needed to avoid slow startup
def lazy_import_components():
    """Lazy import heavy dependencies only when initializing."""
    from rag.db.config import VectorDBConfig
    from rag.db.pinecone_db import PineconeDB
    from rag.bm25.bm25_index import BM25Index
    from rag.query_engine.query_engine import QueryEngine
    from rag.embedding.embedders import MultiModelEmbedder
    from rag.reranker.cross_encoder_reranker import CrossEncoderReranker
    from rag.query_engine.gemini_client import GeminiClient

    return {
        "VectorDBConfig": VectorDBConfig,
        "PineconeDB": PineconeDB,
        "BM25Index": BM25Index,
        "QueryEngine": QueryEngine,
        "MultiModelEmbedder": MultiModelEmbedder,
        "CrossEncoderReranker": CrossEncoderReranker,
        "GeminiClient": GeminiClient,
    }


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global components
query_engine: Optional[Any] = None
initialization_error: Optional[str] = None


def initialize_components():
    """Initialize all RAG components at startup."""
    global query_engine, initialization_error

    try:
        logger.info("=" * 80)
        logger.info("INITIALIZING RAG API SERVICE")
        logger.info("=" * 80)

        # Lazy import heavy dependencies
        logger.info("Loading dependencies...")
        components = lazy_import_components()

        VectorDBConfig = components["VectorDBConfig"]
        PineconeDB = components["PineconeDB"]
        QueryEngine = components["QueryEngine"]
        MultiModelEmbedder = components["MultiModelEmbedder"]
        CrossEncoderReranker = components["CrossEncoderReranker"]
        GeminiClient = components["GeminiClient"]
        logger.info("   ✓ Dependencies loaded")

        # 1. Vector Database (Pinecone)
        logger.info("1/5: Connecting to Pinecone vector database...")
        config = VectorDBConfig()
        vector_db = PineconeDB(config)
        vector_db.connect()

        # Set index
        index_name = os.getenv("PINECONE_INDEX_NAME", "cs431-embeddings")
        vector_db.index = vector_db.client.Index(index_name)
        vector_db.dimension = 1024
        logger.info(f"   ✓ Connected to Pinecone index: {index_name}")

        # 2. BM25 Index
        logger.info("2/5: Loading BM25 index...")
        bm25_path = os.getenv("BM25_INDEX_PATH", "data/prepared/bm25_index.pkl")
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)
        logger.info(f"   ✓ BM25 index loaded from {bm25_path}")

        # 3. Embedder (Local multi-model)
        logger.info("3/5: Initializing embedder...")
        embedder = MultiModelEmbedder()
        logger.info("   ✓ Embedder initialized")

        # 4. Reranker (Local cross-encoder)
        logger.info("4/5: Initializing reranker...")
        reranker = CrossEncoderReranker()
        logger.info("   ✓ Reranker initialized")

        # 5. LLM Client (Gemini via Vertex AI)
        logger.info("5/5: Initializing Gemini client...")
        llm_client = GeminiClient()
        logger.info("   ✓ Gemini client initialized")

        # 6. Query Engine
        logger.info("Creating query engine...")
        query_engine = QueryEngine(
            vector_db=vector_db,
            bm25_index=bm25_index,
            reranker=reranker,
            embedder=embedder,
            gemini_client=llm_client,
        )
        logger.info("   ✓ Query engine ready")

        logger.info("=" * 80)
        logger.info("RAG API SERVICE READY")
        logger.info("=" * 80)

    except Exception as e:
        error_msg = f"Failed to initialize RAG components: {str(e)}"
        logger.error(error_msg, exc_info=True)
        initialization_error = error_msg
        raise


# Don't initialize on startup - wait for first request
# This avoids slow module loading during import
@app.before_request
def ensure_initialized():
    """Ensure components are initialized before handling requests."""
    global query_engine, initialization_error

    if query_engine is None and initialization_error is None:
        try:
            initialize_components()
        except Exception as e:
            logger.error(f"Lazy initialization failed: {e}")


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Returns:
        200: Service is healthy and ready
        503: Service is unhealthy or not ready
    """
    if initialization_error:
        return jsonify({"status": "unhealthy", "error": initialization_error}), 503

    if query_engine is None:
        return (
            jsonify({"status": "unhealthy", "error": "Query engine not initialized"}),
            503,
        )

    return jsonify(
        {
            "status": "healthy",
            "service": "rag-api",
            "components": {
                "vector_db": "connected",
                "bm25": "loaded",
                "reranker": "loaded",
                "embedder": "loaded",
                "gemini": "initialized",
            },
        }
    )


@app.route("/query", methods=["POST"])
def query():
    """
    Process a query through the complete RAG pipeline.

    Request Body:
        {
            "query": "Your question here",
            "video_id": "optional_video_id_filter",
            "embed_model": "vietnamese" | "bge" | "me5" | "all",
            "top_k": 100,
            "rerank_k": 20,
            "context_k": 5
        }

    Response:
        {
            "answer": "Generated answer with citations",
            "contexts": [
                {
                    "chunk_id": "...",
                    "text": "...",
                    "video_id": "...",
                    "start_time": 120.5,
                    "end_time": 150.0,
                    "rerank_score": 0.95
                }
            ],
            "metadata": {
                "query": "...",
                "embed_model": "vietnamese",
                "processing_time_ms": 2500,
                "vector_count": 100,
                "bm25_count": 100,
                "fused_count": 50,
                "reranked_count": 20
            }
        }

    Status Codes:
        200: Success
        400: Invalid request
        500: Processing error
        503: Service not ready
    """
    if query_engine is None:
        return (
            jsonify({"error": "Service not ready", "details": initialization_error}),
            503,
        )

    try:
        # Parse request
        data = request.json
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        query_text = data.get("query", "").strip()
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400

        video_id = data.get("video_id")
        embed_model = data.get("embed_model", "vietnamese")
        # Optional retrieval & rerank parameters (allow UI to tune)
        # Support legacy `top_k` as a convenience for callers
        legacy_top_k = data.get("top_k")
        vector_top_k = int(data.get("vector_top_k", legacy_top_k or 100))
        bm25_top_k = int(data.get("bm25_top_k", legacy_top_k or 100))
        fusion_top_k = int(data.get("fusion_top_k", legacy_top_k or 50))
        # 'context_k' is the UI label for rerank top-K
        rerank_top_k = int(data.get("context_k", data.get("rerank_k", 20)))

        # Start timing
        start_time = time.time()

        # Process query
        logger.info(f"Processing query: {query_text[:100]}... (model: {embed_model})")
        result = query_engine.process_query(
            query_text,
            video_id=video_id,
            embed_model=embed_model,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            fusion_top_k=fusion_top_k,
            rerank_top_k=rerank_top_k,
        )

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        result["metadata"]["processing_time_ms"] = processing_time_ms
        result["metadata"]["embed_model"] = embed_model

        logger.info(f"Query processed in {processing_time_ms}ms")

        return jsonify(result)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        return jsonify({"error": "Query processing failed", "details": str(e)}), 500


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """
    Retrieve relevant chunks without answer generation.

    Request Body:
        {
            "query": "Your question here",
            "video_id": "optional_video_id_filter",
            "top_k": 20
        }

    Response:
        {
            "results": [
                {
                    "chunk_id": "...",
                    "text": "...",
                    "video_id": "...",
                    "start_time": 120.5,
                    "end_time": 150.0,
                    "score": 0.95
                }
            ],
            "metadata": {
                "query": "...",
                "count": 20,
                "processing_time_ms": 500
            }
        }
    """
    if query_engine is None:
        return (
            jsonify({"error": "Service not ready", "details": initialization_error}),
            503,
        )

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        query_text = data.get("query", "").strip()
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400

        video_id = data.get("video_id")
        top_k = data.get("top_k", 20)

        start_time = time.time()

        # Use QueryEngine's retrieve_only method
        result = query_engine.retrieve_only(
            query=query_text, video_id=video_id, top_k=top_k
        )

        processing_time_ms = int((time.time() - start_time) * 1000)
        result["metadata"]["processing_time_ms"] = processing_time_ms

        return jsonify(result)

    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        return jsonify({"error": "Retrieval failed", "details": str(e)}), 500


# TODO: Implement video management and ingestion endpoints in Phase 4
# - POST /videos/upload - Upload and process video
# - GET /videos - List available videos
# - GET /videos/{video_id} - Get video metadata


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return (
        jsonify(
            {
                "error": "Endpoint not found",
                "message": "The requested endpoint does not exist",
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return (
        jsonify(
            {
                "error": "Internal server error",
                "message": "An unexpected error occurred",
            }
        ),
        500,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
