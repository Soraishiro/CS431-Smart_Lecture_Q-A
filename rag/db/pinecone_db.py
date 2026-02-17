"""Pinecone cloud vector database implementation."""

import logging
import time
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

from .interface import VectorDBInterface
from .config import VectorDBConfig

logger = logging.getLogger(__name__)


class PineconeDB(VectorDBInterface):
    """Pinecone cloud vector database implementation."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize Pinecone database client.
        
        Args:
            config: Vector database configuration.
            
        Raises:
            ValueError: If Pinecone API key is not configured.
        """
        if not config.pinecone_api_key:
            raise ValueError("Pinecone API key not configured")
            
        self.config = config
        self.client = None
        self.index = None  # Default index
        self.dimension = None
        self._indexes = {}  # Cache for multiple indexes
        
    def connect(self) -> None:
        """Establish connection to Pinecone cloud.
        
        Raises:
            ConnectionError: If connection fails after retries.
        """
        max_retries = 3
        retry_delays = [1, 2, 4]  # Exponential backoff
        
        for attempt in range(max_retries):
            try:
                self.client = Pinecone(api_key=self.config.pinecone_api_key)
                logger.info("Successfully connected to Pinecone cloud")
                
                # Connect to default index if configured
                if self.config.pinecone_index_name:
                    try:
                        self._get_index(self.config.pinecone_index_name)
                    except Exception as e:
                        logger.warning(f"Could not connect to default index {self.config.pinecone_index_name}: {e}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to connect to Pinecone after {max_retries} attempts")
                    raise ConnectionError(f"Failed to connect to Pinecone: {e}")

    def _get_index(self, index_name: str):
        """Get or initialize connection to a specific index."""
        if index_name in self._indexes:
            return self._indexes[index_name]
            
        if not self.client:
            raise RuntimeError("Pinecone client not connected")
            
        try:
            # Check if index exists
            if index_name not in self.client.list_indexes().names():
                logger.warning(f"Index {index_name} does not exist!")
            
            index = self.client.Index(index_name)
            self._indexes[index_name] = index
            
            # Set as default if first one or matches config
            if self.index is None or index_name == self.config.pinecone_index_name:
                self.index = index
                # Fetch stats to get dimension
                try:
                    stats = index.describe_index_stats()
                    self.dimension = stats.get('dimension')
                except Exception:
                    logger.warning(f"Could not retrieve dimension for index {index_name}")
                    
            return index
        except Exception as e:
            raise RuntimeError(f"Failed to connect to index {index_name}: {e}")

    def create_index(self, index_name: str, dimension: int, metric: str = "cosine") -> None:
        """Create a new index/collection in the vector database."""
        if not self.client:
            raise RuntimeError("Client not connected")
            
        try:
            if index_name in self.client.list_indexes().names():
                logger.info(f"Index {index_name} already exists")
                self._get_index(index_name)
                return

            from pinecone import ServerlessSpec
            # Use default region from config or hardcode for now
            region = "us-east-1" 
            
            logger.info(f"Creating index {index_name} (dim={dimension}, metric={metric})...")
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=region
                )
            )
            
            # Wait for index to be ready
            while not self.client.describe_index(index_name).status.get("ready"):
                time.sleep(1)
            
            self._get_index(index_name)
            logger.info(f"Created Pinecone index {index_name} with dimension {dimension}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create Pinecone index: {e}")
    
    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
        batch_size: int = 100
    ) -> None:
        """Insert or update vectors in batches."""
        if not self.index:
            raise RuntimeError("Index not initialized. Call create_index() first.")
        
        # Batch upsert
        total_vectors = len(vectors)
        
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors")
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
                raise

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        namespace: str = "",
        index_name: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            namespace: Namespace to search in.
            index_name: Optional index name to search (overrides default).
        """
        # Use specified index or default
        index = self._get_index(index_name) if index_name else self.index
        
        if not index:
            raise RuntimeError("No index connected")
            
        try:
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata or {}
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def delete_index(self, index_name: str) -> None:
        """Delete an index/collection."""
        if not self.client:
            raise RuntimeError("Client not connected")
            
        try:
            if index_name in self.client.list_indexes().names():
                self.client.delete_index(index_name)
                if index_name in self._indexes:
                    del self._indexes[index_name]
                logger.info(f"Deleted index {index_name}")
            else:
                logger.warning(f"Index {index_name} does not exist")
        except Exception as e:
            raise RuntimeError(f"Failed to delete index: {e}")
