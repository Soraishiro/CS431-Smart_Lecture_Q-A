"""Streamlit UI for simple RAG Q&A over lecture videos.

Layout:
- Sidebar (Left): Upload video & Keyframe display
- Main column: Chat interface (query box) and answer output
- Right column: Metadata and citations (contexts)

Run:
  pip install streamlit
  streamlit run ui/streamlit_app.py

Note: This UI lazily initializes the QueryEngine. If your environment doesn't have
Milvus or embedder models available, the UI will gracefully fall back to BM25-only retrieval.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARED_DIR = REPO_ROOT / "data/prepared"
UPLOAD_DIR = REPO_ROOT / "data/uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Placeholder for keyframes storage
KEYFRAMES_DIR = REPO_ROOT / "source_system/key_frames"


def get_available_videos() -> List[str]:
    videos = set()
    pkl = PREPARED_DIR / "all_prepared.pkl"
    jsonl = PREPARED_DIR / "all_prepared.jsonl"
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            for r in data:
                videos.add(r.get("video_name", ""))
        except Exception:
            pass
    elif jsonl.exists():
        try:
            for line in open(jsonl, "r", encoding="utf-8"):
                try:
                    import json

                    r = json.loads(line)
                    videos.add(r.get("video_name", ""))
                except Exception:
                    continue
        except Exception:
            pass

    # also include uploaded videos
    for path in UPLOAD_DIR.glob("*"):
        if path.is_file():
            videos.add(path.name)

    vids = sorted([v for v in videos if v])
    return vids


def get_api_client():
    """Get API client for backend service."""
    api_url = os.getenv("RAG_API_URL", "http://localhost:8080")
    return api_url


def check_api_health(api_url: str) -> bool:
    """Check if API is healthy."""
    try:
        import requests

        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_api_from_state() -> Optional[str]:
    """Get API URL from session state."""
    if "api_url" not in st.session_state:
        st.session_state.api_url = get_api_client()
    return st.session_state.api_url


def save_uploaded_video(uploaded):
    if uploaded is None:
        return None
    path = UPLOAD_DIR / uploaded.name
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def trigger_backend_processing(video_path: Path, api_url: str):
    """
    Trigger backend API routes for ASR and Keyframe extraction.
    (Currently placeholders/mocks as requested)
    """
    import requests

    # Mock calling /asr
    try:
        # In a real scenario: requests.post(f"{api_url}/asr", files={'file': open(video_path, 'rb')})
        st.toast(f"Triggered /asr for {video_path.name} (Mock)", icon="🎙️")
    except Exception as e:
        st.error(f"Failed to trigger ASR: {e}")

    # Mock calling /keyframe_extract
    try:
        # In a real scenario: requests.post(f"{api_url}/keyframe_extract", json={'video_path': str(video_path)})
        st.toast(f"Triggered /keyframe_extract for {video_path.name} (Mock)", icon="🖼️")
    except Exception as e:
        st.error(f"Failed to trigger Keyframe Extraction: {e}")


def find_keyframe_image(video_id: str, timestamp: float) -> Optional[Path]:
    """
    Find the closest keyframe image for a given video and timestamp.
    Assumes keyframes are stored in source_system/key_frames/{video_id}/frame_{seconds}.jpg
    or similar structure.
    """
    if not KEYFRAMES_DIR.exists():
        return None

    # Try to find a folder for the video
    video_frame_dir = KEYFRAMES_DIR / video_id
    if not video_frame_dir.exists():
        # Fallback: maybe they are all in the root keyframes dir?
        video_frame_dir = KEYFRAMES_DIR

    # Simple matching logic: look for frame closest to timestamp
    # Assuming filenames like "frame_10.jpg" or "{video_id}_10.jpg"
    # For now, let's just try to find an exact match or close match if we knew the naming convention.
    # Since we don't know the exact convention created by the future unzipping,
    # we will look for files containing the video_id and try to parse timestamp.
    
    # Placeholder logic: return None if no obvious match found
    # In a real implementation, we would list files, parse timestamps from filenames, and find closest.
    return None


def app():
    st.set_page_config(layout="wide", page_title="CS431 Q&A UI")
    st.title("CS431: Video Q&A — RAG System")

    # Check API health
    api_url = get_api_from_state()
    api_healthy = check_api_health(api_url)

    if not api_healthy:
        st.error(
            f"⚠️ Backend API not available at {api_url}. Please start the API service."
        )
        st.code(f"python services/api/app.py", language="bash")
        return

    # --- SIDEBAR (Left Panel) ---
    with st.sidebar:
        st.header("Resource Upload")
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file:
            if st.button("Process Video"):
                with st.spinner("Uploading and triggering processing..."):
                    saved_path = save_uploaded_video(uploaded_file)
                    if saved_path:
                        trigger_backend_processing(saved_path, api_url)
                        st.success(f"Uploaded {uploaded_file.name} successfully!")
                    else:
                        st.error("Failed to save file.")

        st.divider()
        
        st.header("Keyframes")
        st.caption("Keyframes from relevant contexts will appear here.")
        
        # Display keyframes from the latest query results
        if "chat_history" in st.session_state and st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            contexts = latest.get("contexts", [])
            
            if contexts:
                # Show keyframes for top 3 contexts to avoid clutter
                for i, ctx in enumerate(contexts[:3]):
                    meta = ctx.get("metadata", {})
                    video_id = meta.get("video_id", "Unknown")
                    start_time = meta.get("start_time", 0.0)
                    
                    st.markdown(f"**Context {i+1}** ({video_id} @ {start_time:.1f}s)")
                    
                    # Attempt to find and display image
                    # For now, since we don't have the files, we show a placeholder or the path we looked for.
                    # In the future, use find_keyframe_image(video_id, start_time)
                    
                    # Mock display for demonstration if file doesn't exist
                    st.info(f"Keyframe for {video_id} at {start_time}s would be displayed here.")
                    # st.image(image_path) # Uncomment when files exist
            else:
                st.text("No contexts found.")
        else:
            st.text("Ask a question to see keyframes.")


    # --- MAIN LAYOUT ---
    # Columns: main (3), right (1)
    col_main, col3 = st.columns([3, 1])

    # No upload panel in main area anymore: default to searching across all videos
    selected_video = "(all)"

    # Middle: Chat
    with col_main:
        st.header("Ask a question")
        query_text = st.text_input("Enter your question here", key="query_input")

        # Model selection
        embed_model = st.selectbox(
            "Embedding model",
            ["vietnamese", "bge", "me5", "all"],
            index=0,
            help="Choose embedding model: vietnamese (768-dim), bge (1024-dim), me5 (1024-dim), or all (multi-model fusion)",
        )

        top_k = st.slider("Top results to display", min_value=1, max_value=20, value=5)

        # Advanced retrieval settings (optional): allow user to tune retrieval sizes
        with st.expander("Advanced settings (tweak retrieval sizes)", expanded=False):
            # Vector search top_k per embedding model (used for vector runs)
            vector_top_k = st.number_input(
                "Vector retrieval top-k per model",
                min_value=1,
                max_value=200,
                value=50,
                step=1,
                help="Number of vector search results to retrieve per model (for multi-model, per model collection).",
            )
            bm25_top_k = st.number_input(
                "BM25 retrieval top-k",
                min_value=1,
                max_value=200,
                value=50,
                step=1,
                help="Number of BM25 results to retrieve.",
            )
            fusion_top_k = st.number_input(
                "Fusion top-k",
                min_value=1,
                max_value=200,
                value=50,
                step=1,
                help="Number of results after RRF fusion",
            )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if st.button("Send") and query_text:
            with st.spinner("Processing query..."):
                try:
                    import requests

                    # Build request
                    payload = {
                        "query": query_text,
                        "embed_model": embed_model,
                        "context_k": top_k,
                        "vector_top_k": int(vector_top_k),
                        "bm25_top_k": int(bm25_top_k),
                        "fusion_top_k": int(fusion_top_k),
                    }

                    if selected_video != "(all)":
                        payload["video_id"] = selected_video

                    # Call API
                    response = requests.post(
                        f"{api_url}/query", json=payload, timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "")
                        contexts = result.get("contexts", [])
                        metadata = result.get("metadata", {})

                        st.session_state.chat_history.append(
                            {
                                "q": query_text,
                                "a": answer,
                                "contexts": contexts,
                                "metadata": metadata,
                                "request_params": {
                                    "embed_model": embed_model,
                                    "vector_top_k": int(vector_top_k),
                                    "bm25_top_k": int(bm25_top_k),
                                    "fusion_top_k": int(fusion_top_k),
                                    "context_k": top_k,
                                },
                            }
                        )
                    else:
                        error_msg = response.json().get("error", "Unknown error")
                        st.error(f"Query failed: {error_msg}")
                        st.session_state.chat_history.append(
                            {
                                "q": query_text,
                                "a": f"ERROR: {error_msg}",
                                "contexts": [],
                                "metadata": {},
                                "request_params": {
                                    "embed_model": embed_model,
                                    "vector_top_k": int(vector_top_k),
                                    "bm25_top_k": int(bm25_top_k),
                                    "fusion_top_k": int(fusion_top_k),
                                    "context_k": top_k,
                                },
                            }
                        )

                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.session_state.chat_history.append(
                        {
                            "q": query_text,
                            "a": f"ERROR: {e}",
                            "contexts": [],
                            "metadata": {},
                            "request_params": {
                                "embed_model": embed_model,
                                "vector_top_k": int(vector_top_k),
                                "bm25_top_k": int(bm25_top_k),
                                "fusion_top_k": int(fusion_top_k),
                                "context_k": top_k,
                            },
                        }
                    )

        # Show chat history
        for item in reversed(st.session_state.chat_history[-10:]):
            st.markdown("**Q:** " + item["q"])
            st.markdown("**A:** " + item["a"])

            # Show metadata if available
            if item.get("metadata"):
                meta = item["metadata"]
                st.caption(
                    f"⏱️ {meta.get('processing_time_ms', 0)}ms | "
                    f"📊 {meta.get('reranked_count', 0)} contexts"
                )

            st.markdown("---")

    # Right: Metadata and citations
    with col3:
        st.header("Metadata & Citations")
        if "chat_history" in st.session_state and st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            st.markdown("**Latest query:** " + latest["q"])
            st.markdown("**Contexts (top):**")
            contexts = latest.get("contexts", [])
            if contexts:
                for c in contexts[:top_k]:
                    meta = (
                        c.get("metadata", {})
                        if isinstance(c.get("metadata"), dict)
                        else c
                    )

                    exp = st.expander(
                        f"{meta.get('chunk_id', meta.get('id', 'unknown'))} — {meta.get('video_id', '')}"
                    )
                    with exp:
                        st.markdown(f"**Video**: {meta.get('video_id', '')}")
                        st.markdown(
                            f"**Time**: {meta.get('start_time', 0.0)}s - {meta.get('end_time', 0.0)}s"
                        )
                        if "rerank_score" in meta:
                            st.markdown(
                                f"**Score**: {meta.get('rerank_score', 0.0):.4f}"
                            )
                        st.markdown(
                            f"**Text**: {meta.get('text', '')[:350]}{'...' if len(meta.get('text', ''))>350 else ''}"
                        )
            else:
                st.markdown("No contexts yet — submit a query first.")
            # Show the retrieval settings used for the latest query
            req = latest.get("request_params", {})
            if req:
                st.caption(
                    f"⚙ model={req.get('embed_model')} | vector_top_k={req.get('vector_top_k')} | bm25_top_k={req.get('bm25_top_k')} | fusion_top_k={req.get('fusion_top_k')} | context_k={req.get('context_k')}"
                )
        else:
            st.markdown("No queries yet — ask a question in the center panel.")


if __name__ == "__main__":
    app()
