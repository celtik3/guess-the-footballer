from typing import Dict, Any
import numpy as np

# from .inference import get_model_bundle
from .embedding_store import EmbeddingStore

_STORE = None

def get_store() -> EmbeddingStore:
    global _STORE
    if _STORE is None:
        _STORE = EmbeddingStore.load()
    return _STORE


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors.
    Returns a value in [-1, 1], where 1 = most similar.
    """
    ## I added small epsilon to avoid division by zero.
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 1D vectors.
    Smaller = more similar.
    """
    return float(np.linalg.norm(a - b))


def player_similarity(features_a: Dict[str, Any], features_b: Dict[str, Any]) -> Dict[str, float]:
    """
    Given two players' feature dicts, compute similarity metrics.
    Returns a dict with:
      - cosine_sim: higher is more similar
      - euclidean_dist: lower is more similar
    Uses cached embeddings (precomputed) by player_id.
    """
    store = get_store()

    pid_a = int(features_a["player_id"])
    pid_b = int(features_b["player_id"])

    emb_a = store.vec(pid_a)
    emb_b = store.vec(pid_b)

    return {
        "cosine_sim": cosine_similarity(emb_a, emb_b),
        "euclidean_dist": euclidean_distance(emb_a, emb_b),
    }