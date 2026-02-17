"""Milvus localhost vector database implementation."""

import logging
import time
from typing import List, Dict, Any
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from ..interface import VectorDBInterface
from ..config import VectorDBConfig

logger = logging.getLogger(__name__)


class MilvusDB(VectorDBInterface):
    """Milvus localhost vector database implementation."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize Milvus database client.
        
        Args:
            config: Vector database configuration.
        """
        self.config = config
        self.collection = None
        self.dimension = None
        self.collection_name = None
        
    def connect(self) -> None:
        """Establish connection to Milvus localhost.
        
        Raises:
            ConnectionError: If connection fails after retries.
        """
        max_retries = 3
        retry_delays = [1, 2, 4]  # Exponential backoff
        
        for attempt in range(max_retries):
            try:
                connections.connect(
                    alias="default",
                    host=self.config.milvus_host,
                    port=self.config.milvus_port,
                    user=self.config.milvus_user if self.config.milvus_user else None,
                    password=self.config.milvus_password if self.config.milvus_password else None
                )
                logger.info(
                    f"Successfully connected to Milvus at "
                    f"{self.config.milvus_host}:{self.config.milvus_port}"
                )
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.warning(
                        f"Milvus connection attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to Milvus after {max_retries} attempts: {e}"
                    )
    
    def create_index(self, index_name: str, dimension: int, metric: str = "cosine") -> None:
        """Create a Milvus collection with IVF_FLAT index.
        
        Args:
            index_name: Name of the collection to create.
            dimension: Dimensionality of vectors (e.g., 1024 for BGE).
            metric: Distance metric ("cosine", "L2", or "IP").
            
        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If collection creation fails.
        """
        if dimension <= 0:
            raise ValueError(f"Invalid dimension: {dimension}")
        
        # Map metric names
        metric_map = {
            "cosine": "COSINE",
            "euclidean": "L2",
            "l2": "L2",
            "dotproduct": "IP",
            "ip": "IP"
        }
        milvus_metric = metric_map.get(metric.lower())
        if not milvus_metric:
            raise ValueError(f"Invalid metric: {metric}")
        
        try:
            # Check if collection already exists
            if utility.has_collection(index_name):
                logger.info(f"Collection {index_name} already exists")
                self.collection = Collection(index_name)
                self.collection.load()
                self.dimension = dimension
                self.collection_name = index_name
                return
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="start_time", dtype=DataType.FLOAT),
                FieldSchema(name="end_time", dtype=DataType.FLOAT),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="enhanced", dtype=DataType.VARCHAR, max_length=65535),  # LLM-enhanced text
            ]
            schema = CollectionSchema(fields=fields, description="Lecture embeddings collection")
            
            # Create collection
            self.collection = Collection(name=index_name, schema=schema)
            
            # Create IVF_FLAT index
            index_params = {
                "metric_type": milvus_metric,
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
            # Load collection
            self.collection.load()
            
            self.dimension = dimension
            self.collection_name = index_name
            logger.info(f"Created Milvus collection {index_name} with dimension {dimension}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create Milvus collection: {e}")
    
    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
        batch_size: int = 100
    ) -> None:
        """Insert or update vectors in batches.
        
        Args:
            vectors: List of dicts with 'id', 'values', and 'metadata' keys.
            namespace: Partition name (optional, not used in basic implementation).
            batch_size: Number of vectors per batch (default: 100).
            
        Raises:
            ValueError: If vector dimensions don't match collection dimensions.
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_index() first.")
        
        # Validate dimensions
        for vec in vectors:
            if len(vec.get("values", [])) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vec['values'])} doesn't match "
                    f"collection dimension {self.dimension}"
                )
        
        # Prepare data for insertion
        total_vectors = len(vectors)
        failed_chunks = []
        
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                # Extract fields
                ids = [vec["id"] for vec in batch]
                embeddings = [vec["values"] for vec in batch]
                chunk_ids = [vec.get("metadata", {}).get("chunk_id", "") for vec in batch]
                video_ids = [vec.get("metadata", {}).get("video_id", "") for vec in batch]
                start_times = [vec.get("metadata", {}).get("start_time", 0.0) for vec in batch]
                end_times = [vec.get("metadata", {}).get("end_time", 0.0) for vec in batch]
                texts = [vec.get("metadata", {}).get("text", "") for vec in batch]
                enhanced_texts = [vec.get("metadata", {}).get("enhanced", "") for vec in batch]
                
                # Insert data
                data = [ids, embeddings, chunk_ids, video_ids, start_times, end_times, texts, enhanced_texts]
                self.collection.insert(data)
                
                logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors")
            except Exception as e:
                logger.error(f"Failed to upsert batch starting at index {i}: {e}")
                failed_chunks.extend([vec.get("id") for vec in batch])
        
        # Flush to persist data
        self.collection.flush()
        
        if failed_chunks:
            logger.warning(f"Failed to upsert {len(failed_chunks)} chunks: {failed_chunks}")
        else:
            logger.info(f"Successfully upserted {total_vectors} vectors")
    
    def search(
        self,
        query_vector: List[float],
        top_k: int,
        namespace: str = ""
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            namespace: Partition to search in (optional).
            
        Returns:
            List of dicts with 'id', 'score', and 'metadata' keys.
            
        Raises:
            ValueError: If query vector dimension is invalid.
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_index() first.")
        
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} doesn't match "
                f"collection dimension {self.dimension}"
            )
        
        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            start_time = time.time()
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["chunk_id", "video_id", "start_time", "end_time", "text", "enhanced"]
            )
            latency_ms = (time.time() - start_time) * 1000
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.id,
                        "score": hit.score,
                        "metadata": {
                            "chunk_id": hit.entity.get("chunk_id"),
                            "video_id": hit.entity.get("video_id"),
                            "start_time": hit.entity.get("start_time"),
                            "end_time": hit.entity.get("end_time"),
                            "text": hit.entity.get("text"),
                            "enhanced": hit.entity.get("enhanced", "")  # LLM-enhanced text for context
                        }
                    })
            
            logger.info(
                f"Milvus search returned {len(formatted_results)} results "
                f"in {latency_ms:.2f}ms"
            )
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_index(self, index_name: str) -> None:
        """Delete a Milvus collection.
        
        Args:
            index_name: Name of the collection to delete.
            
        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            if utility.has_collection(index_name):
                utility.drop_collection(index_name)
                logger.info(f"Deleted Milvus collection {index_name}")
                self.collection = None
                self.dimension = None
                self.collection_name = None
            else:
                logger.warning(f"Collection {index_name} does not exist")
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection {index_name}: {e}")
