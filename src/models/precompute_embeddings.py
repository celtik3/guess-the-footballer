import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.mlp_model import MLP


## Paths to be used.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts_full"

DATA_PATH = PROC_DIR / "players_full.csv"
MODEL_PATH = ARTIFACTS_DIR / "mlp_full.pt"
PREPROC_PATH = ARTIFACTS_DIR / "preproc_full.pkl"

OUT_EMB_PATH = ARTIFACTS_DIR / "embeddings.npy"
OUT_INDEX_PATH = ARTIFACTS_DIR / "player_index.csv"


## Loading artifacts.

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = MLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    ## I am using the same feature columns used during training.
    feature_cols = [
        "age_band",
        "nation_region",
        "league_region",
        "value_bin",
        "style",
        "current_club_domestic_competition_id",
        "age",
        "total_minutes",
        "games_played",
        "starts",
        "starts_pct",
        "goals",
        "assists",
        "goals_per90",
        "assists_per90",
        "latest_value_eur",
        "peak_value_eur",
        "peak_year",
    ]

    X = df[feature_cols]

    print("Loading preprocessor...")
    with open(PREPROC_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    X_proc = preprocessor.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    X_tensor = torch.tensor(X_proc, dtype=torch.float32)

    print("Loading model...")
    model = load_model()

    ## Extracting embeddings.
    print("Computing embeddings...")
    with torch.no_grad():
        embeddings = model.get_embeddings(X_tensor)
    embeddings_np = embeddings.numpy()

    print(f"Embeddings shape: {embeddings_np.shape}")

    ## Saving embeddings and index mapping.
    np.save(OUT_EMB_PATH, embeddings_np)

    index_df = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "player_id": df["player_id"],
        "player_name": df["player_name"],
    })
    index_df.to_csv(OUT_INDEX_PATH, index=False)

    print("Saved:")
    print(f" - {OUT_EMB_PATH}")
    print(f" - {OUT_INDEX_PATH}")


if __name__ == "__main__":
    main()