"""
Multi-model embedding pipeline for Vietnamese lecture transcripts.

Supports 3 embedding models in parallel:
1. BGE-M3 (multilingual, SOTA semantic)
2. Vietnamese-native (dangvantuan/vietnamese-document-embedding)
3. Multilingual-e5-instruct (intfloat/multilingual-e5-large-instruct)
"""

from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


@dataclass
class EmbeddingResult:
    """Container for multi-model embeddings"""
    text: str
    bge_embedding: np.ndarray  # 1024-dim
    vietnamese_embedding: np.ndarray  # 768-dim
    me5_embedding: np.ndarray  # 768-dim
    metadata: Optional[Dict] = None


class MultiModelEmbedder:
    """
    Parallel embedding generation using 3 models.
    
    Models:
    - BGE-M3: BAAI/bge-m3 (1024-dim)
    - Vietnamese: dangvantuan/vietnamese-document-embedding (768-dim)
    - me5: Alibaba-NLP/me5-multilingual-base (768-dim)
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize all 3 embedding models.
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"Loading models on device: {device}")
        
        # Model 1: BGE-M3 (multilingual, SOTA)
        print("Loading BGE-M3...")
        self.model_bge = SentenceTransformer('BAAI/bge-m3', device=device)
        
        # Model 2: Vietnamese-native
        print("Loading Vietnamese-native embedder...")
        self.model_vietnamese = SentenceTransformer(
            'dangvantuan/vietnamese-document-embedding',
            device=device
        )
        
        # Model 3: ME5-multilingual
        print("Loading ME5-multilingual...")
        self.model_me5 = SentenceTransformer(
            'intfloat/multilingual-e5-large-instruct',
            device=device
        )
        
        print("All models loaded successfully!")
    
    def embed_single(self, text: str, metadata: Optional[Dict] = None) -> EmbeddingResult:
        """
        Generate embeddings for a single text using all 3 models.
        
        Args:
            text: Input text to embed
            metadata: Optional metadata to attach
            
        Returns:
            EmbeddingResult with all 3 embeddings
        """
        # Generate embeddings from all models
        bge_emb = self.model_bge.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        viet_emb = self.model_vietnamese.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        me5_emb = self.model_me5.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        
        return EmbeddingResult(
            text=text,
            bge_embedding=bge_emb,
            vietnamese_embedding=viet_emb,
            me5_embedding=me5_emb,
            metadata=metadata
        )
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        metadata_list: Optional[List[Dict]] = None
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            metadata_list: Optional list of metadata dicts (same length as texts)
            
        Returns:
            List of EmbeddingResult objects
        """
        if metadata_list is None:
            metadata_list = [None] * len(texts)
        
        # Encode all texts with each model
        print(f"Encoding {len(texts)} texts with BGE-M3...")
        bge_embeddings = self.model_bge.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"Encoding {len(texts)} texts with Vietnamese embedder...")
        viet_embeddings = self.model_vietnamese.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"Encoding {len(texts)} texts with me5-multilingual...")
        me5_embeddings = self.model_me5.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Combine results
        results = []
        for i, text in enumerate(texts):
            results.append(EmbeddingResult(
                text=text,
                bge_embedding=bge_embeddings[i],
                vietnamese_embedding=viet_embeddings[i],
                me5_embedding=me5_embeddings[i],
                metadata=metadata_list[i]
            ))
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'bge': {
                'name': 'BAAI/bge-m3',
                'dims': 1024,
                'description': 'Multilingual, SOTA semantic understanding'
            },
            'vietnamese': {
                'name': 'dangvantuan/vietnamese-document-embedding',
                'dims': 768,
                'description': 'Vietnamese-native embeddings'
            },
            'me5': {
                'name': 'Alibaba-NLP/me5-multilingual-base',
                'dims': 768,
                'description': 'Multilingual embeddings with strong technical term support'
            }
        }


def test_embedder():
    """Quick test of the embedder"""
    embedder = MultiModelEmbedder(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test single embedding
    test_text = "Mô hình học máy sử dụng gradient descent để tối ưu hóa hàm loss."
    result = embedder.embed_single(test_text)
    
    print("\nTest Results:")
    print(f"Text: {result.text}")
    print(f"BGE embedding shape: {result.bge_embedding.shape}")
    print(f"Vietnamese embedding shape: {result.vietnamese_embedding.shape}")
    print(f"me5 embedding shape: {result.me5_embedding.shape}")
    
    # Test batch embedding
    test_texts = [
        "Neural network là gì?",
        "Transformer architecture sử dụng self-attention mechanism.",
        "Gradient descent là thuật toán tối ưu hóa."
    ]
    
    results = embedder.embed_batch(test_texts, batch_size=2)
    print(f"\nBatch embedding completed: {len(results)} texts")


if __name__ == '__main__':
    test_embedder()
