"""
retriever.py — MediVision RAG retrieval
Builds a FAISS index from extracted ViT features and retrieves
the most similar X-ray reports for a given query image.

Usage:
    python src/retriever.py
"""

import os
import torch
import numpy as np
import pandas as pd
import faiss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_PT  = os.path.join("data", "features.pt")
DATASET_CSV  = os.path.join("data", "dataset_index.csv")
INDEX_OUT    = os.path.join("data", "faiss_index.bin")
MAPPING_OUT  = os.path.join("data", "index_mapping.csv")

TOP_K = 5        # number of similar reports to retrieve
FEATURE_DIM = 768


# ---------------------------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------------------------
def build_index(features: dict) -> tuple:
    """
    Build a FAISS flat L2 index from the feature vectors.
    Returns (index, list of image_ids in index order).
    """
    image_ids = list(features.keys())
    vectors = np.stack([features[i].numpy() for i in image_ids]).astype("float32")

    # Normalize vectors so L2 distance ≈ cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(FEATURE_DIM)   # Inner Product on normalized = cosine
    index.add(vectors)

    print(f"FAISS index built — {index.ntotal} vectors, dimension {FEATURE_DIM}")
    return index, image_ids


def save_index(index, image_ids: list):
    """Save FAISS index and image_id order mapping to disk."""
    faiss.write_index(index, INDEX_OUT)
    pd.DataFrame({"image_id": image_ids}).to_csv(MAPPING_OUT, index=False)
    print(f"Saved index to {INDEX_OUT}")
    print(f"Saved mapping to {MAPPING_OUT}")


def load_index() -> tuple:
    """Load FAISS index and image_id mapping from disk."""
    index = faiss.read_index(INDEX_OUT)
    image_ids = pd.read_csv(MAPPING_OUT)["image_id"].tolist()
    return index, image_ids


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_similar_reports(
    query_vector: torch.Tensor,
    index,
    image_ids: list,
    df: pd.DataFrame,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Given a query ViT feature vector, find the top_k most similar
    X-rays and return their report text.

    Returns a list of dicts with keys: image_id, findings, impression, score.
    """
    query = query_vector.numpy().astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)

    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        image_id = image_ids[idx]
        row = df[df["image_id"] == image_id]
        if row.empty:
            continue
        results.append({
            "image_id": image_id,
            "findings":   row.iloc[0]["findings"],
            "impression": row.iloc[0]["impression"],
            "score":      float(score),
        })

    return results


def format_rag_context(results: list[dict]) -> str:
    """
    Format retrieved reports into a text block to pass to Gemini as context.
    """
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"Reference Report {i} (similarity: {r['score']:.3f}):\n"
            f"Findings: {r['findings']}\n"
            f"Impression: {r['impression']}"
        )
    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Main — build and test the index
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(FEATURES_PT):
        raise SystemExit("data/features.pt not found. Run extractor.py first.")

    print("Loading feature vectors...")
    features = torch.load(FEATURES_PT, weights_only=True)
    print(f"Loaded {len(features)} feature vectors")

    df = pd.read_csv(DATASET_CSV)

    index, image_ids = build_index(features)
    save_index(index, image_ids)

    # Sanity check — use the first image as a query and retrieve similar ones
    print("\n--- Retrieval Test ---")
    sample_id = image_ids[0]
    query_vec = features[sample_id]
    results = retrieve_similar_reports(query_vec, index, image_ids, df, top_k=3)

    for r in results:
        print(f"\nimage_id : {r['image_id']}")
        print(f"score    : {r['score']:.4f}")
        print(f"impression: {r['impression'][:120]}...")


if __name__ == "__main__":
    main()