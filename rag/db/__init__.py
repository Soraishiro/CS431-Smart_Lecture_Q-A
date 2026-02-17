"""Vector database abstraction layer for RAG system.

This module avoids importing optional DB adapters (like Pinecone) at
import-time so code that only needs Milvus doesn't fail if Pinecone
client libraries are not installed.
"""

from .interface import VectorDBInterface

# Import adapters lazily / guarded so missing optional deps don't break
# simple flows that only use Milvus.
try:
    from .pinecone_db import PineconeDB
except Exception:
    PineconeDB = None

try:
    from .milvus_db import MilvusDB
except Exception:
    MilvusDB = None

__all__ = ["VectorDBInterface", "PineconeDB", "MilvusDB"]
