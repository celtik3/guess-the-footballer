from typing import Dict, Any

import numpy as np

from .inference import get_model_bundle


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors.
    Returns a value in [-1, 1], where 1 = most similar.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 1D vectors.
    Smaller = more similar.
    """
    return float(np.linalg.norm(a - b))


def player_similarity(
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
) -> Dict[str, float]:
    """
    Given two players' feature dicts, compute similarity metrics.

    Returns a dict with:
      - cosine_sim: higher is more similar
      - euclidean_dist: lower is more similar
    """
    bundle = get_model_bundle()
    emb_a = bundle.get_embedding(features_a)
    emb_b = bundle.get_embedding(features_b)

    cos_sim = cosine_similarity(emb_a, emb_b)
    euc_dist = euclidean_distance(emb_a, emb_b)

    return {
        "cosine_sim": cos_sim,
        "euclidean_dist": euc_dist,
    }