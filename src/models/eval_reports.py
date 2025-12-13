import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.models.mlp_model import MLP


RANDOM_STATE = 42


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_artifacts(artifacts_dir: Path):
    ckpt_path = artifacts_dir / "mlp_full.pt"
    preproc_path = artifacts_dir / "preproc_full.pkl"
    le_path = artifacts_dir / "label_encoder_full.pkl"

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(preproc_path, "rb") as f:
        preproc = pickle.load(f)

    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)

    model = MLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, preproc, label_encoder


def get_feature_cols():
    ## This must match my training/precompute scripts.
    cat_cols = [
        "age_band",
        "nation_region",
        "league_region",
        "value_bin",
        "style",
        "current_club_domestic_competition_id",
    ]
    num_cols = [
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
    return cat_cols + num_cols


def make_confusion_matrix_png(df: pd.DataFrame, model, preproc, label_encoder, out_path: Path):
    feature_cols = get_feature_cols()
    X = df[feature_cols].copy()
    y = df["position_group"].astype(str).copy()

    ## Training did label_encoder.fit_transform(y)
    y_enc = label_encoder.transform(y)

    ## Training did preprocessor.fit_transform(X) then split.
    ## So, I used the saved fitted preprocessor and only transform.
    X_proc = preproc.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    X_proc = np.asarray(X_proc, dtype=np.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_proc,
        y_enc,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    cm = confusion_matrix(y_test, preds, labels=np.arange(len(label_encoder.classes_)))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)

    plt.figure(figsize=(9, 7))
    disp.plot(xticks_rotation=30, values_format="d")
    plt.title("Confusion Matrix (Position Group)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_embedding_png(df: pd.DataFrame, artifacts_dir: Path, out_path: Path):
    ##I am using the pre-computed embeddings.npy file.
    emb_path = artifacts_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} not found. Run: python -m src.models.precompute_embeddings"
        )

    emb = np.load(emb_path)  ## shape: (N, hidden_dim)
    labels = df["position_group"].astype(str).values

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb_2d = pca.fit_transform(emb)

    ## Mapping labels to ints for coloring.
    uniq = sorted(set(labels))
    label_to_id = {lab: i for i, lab in enumerate(uniq)}
    c = np.array([label_to_id[x] for x in labels])

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=c, s=8)
    plt.title("MLP Embeddings (PCA, colored by position_group)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    ## Simple legend.
    cmap = scatter.get_cmap()
    num_labels = len(uniq)
    colors = [cmap(i) for i in np.linspace(0, 1, num_labels)]
    handles = [
        plt.Line2D([], [], linestyle="", marker="o", label=lab, color=colors[label_to_id[lab]])
        for lab in uniq
    ]
    plt.legend(handles=handles, title="position_group", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="data/processed/players_full.csv")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts_full")
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    data_csv = Path(args.data_csv)
    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(data_csv)

    model, preproc, label_encoder = load_artifacts(artifacts_dir)

    cm_path = out_dir / "confusion_matrix.png"
    emb_path = out_dir / "embeddings_pca.png"

    make_confusion_matrix_png(df, model, preproc, label_encoder, cm_path)
    make_embedding_png(df, artifacts_dir, emb_path)

    print("Saved:")
    print(" -", cm_path)
    print(" -", emb_path)


if __name__ == "__main__":
    main()