"""Abstract base class for vector database interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class VectorDBInterface(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the vector database.
        
        Raises:
            ConnectionError: If connection fails after retries.
        """
        pass

    @abstractmethod
    def create_index(self, index_name: str, dimension: int, metric: str = "cosine") -> None:
        """Create a new index/collection in the vector database.
        
        Args:
            index_name: Name of the index/collection to create.
            dimension: Dimensionality of the vectors.
            metric: Distance metric to use (default: "cosine").
            
        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If index creation fails.
        """
        pass

    @abstractmethod
    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
        batch_size: int = 100
    ) -> None:
        """Insert or update vectors with metadata in batches.
        
        Args:
            vectors: List of dicts with 'id', 'values', and 'metadata' keys.
            namespace: Namespace/partition for the vectors.
            batch_size: Number of vectors to upsert per batch.
            
        Raises:
            ValueError: If vector dimensions don't match index dimensions.
        """
        pass

    @abstractmethod
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
            namespace: Namespace/partition to search in.
            
        Returns:
            List of dicts with 'id', 'score', and 'metadata' keys.
            
        Raises:
            ValueError: If query vector dimension is invalid.
        """
        pass

    @abstractmethod
    def delete_index(self, index_name: str) -> None:
        """Delete an index/collection.
        
        Args:
            index_name: Name of the index/collection to delete.
            
        Raises:
            RuntimeError: If deletion fails.
        """
        pass
