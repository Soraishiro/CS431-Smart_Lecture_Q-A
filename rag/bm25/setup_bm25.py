"""Build BM25 index using Underthesea tokenizer.

This script initializes the BM25 index using the `underthesea` library for Vietnamese tokenization.
It reads documents from `data/prepared/bm25_docs.jsonl` and saves the built index to `data/prepared/bm25_index.pkl`.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

# Add project root to sys.path to allow importing 'rag'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from rag.bm25.bm25_index import BM25Index


def build_and_save_index():
    # Use Underthesea tokenizer for indexing (lighter, pure-Python).
    try:
        import underthesea  # noqa: F401
        print("Using Underthesea for tokenization")
    except Exception:
        print("Underthesea not installed. Install with: pip install underthesea")
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=SyntaxWarning, module=r"underthesea.*"
                )
                import underthesea  # noqa: F401
            print("Using Underthesea for tokenization")
        except Exception:
            print("Underthesea not installed. Install with: pip install underthesea")
            return

    # load docs (each doc must have 'chunk_id' and 'text')
    docs = []
    input_path = "data/prepared/bm25_docs.jsonl"
    print(f"Loading documents from {input_path}...")
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
        print(f"Loaded {len(docs)} documents.")
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Run prepare_embeddings.py first.")
        return

    # index using Underthesea tokenization
    print("Building BM25 index...")
    bm25 = BM25Index(k1=1.5, b=0.75, use_vncorenlp=False, use_underthesea=True)
    bm25.index_documents(docs)

    # pickle for fast reuse
    os.makedirs("data/prepared", exist_ok=True)
    output_path = "data/prepared/bm25_index.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"BM25 built with Underthesea and saved to {output_path}")


if __name__ == "__main__":
    build_and_save_index()
