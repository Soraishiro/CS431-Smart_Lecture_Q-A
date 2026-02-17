"""Configuration loading for vector databases."""

import os
from typing import Optional

class VectorDBConfig:
    """Configuration for vector database connections."""
    
    def __init__(self):
        """Load configuration from environment variables and GCP Secret Manager."""
        self.provider = os.getenv("VECTOR_DB_PROVIDER", "pinecone")
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID")
        
        # Pinecone configuration
        self.pinecone_api_key = self._get_pinecone_api_key()
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "cs431-embeddings")
        
        # Milvus configuration
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        self.milvus_user = os.getenv("MILVUS_USER", "")
        self.milvus_password = os.getenv("MILVUS_PASSWORD", "")
        self.milvus_collection = os.getenv("MILVUS_COLLECTION", "cs431-embeddings")
        
    def _get_pinecone_api_key(self) -> Optional[str]:
        """Get Pinecone API key from Secret Manager or environment variable.
        
        Returns:
            Pinecone API key or None if not configured.
        """
        # Try environment variable first
        api_key = os.getenv("PINECONE_API_KEY")
        if api_key:
            return api_key
                
        return None
