import os
from pathlib import Path
import pickle

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp_model import MLP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts_full"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROC_DIR / "players_full.csv"


EXPERIMENT_NAME = "mlp_full_players"
RANDOM_STATE = 42
HIDDEN_DIM = 128
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-4


## Helpers for data loading and preprocessing.

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    ## Features and target.
    target_col = "position_group"

    ## Categorical features.
    cat_cols = [
        "age_band",
        "nation_region",
        "league_region",
        "value_bin",
        "style",
        "current_club_domestic_competition_id",
    ]

    ## Numeric features.
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

    feature_cols = cat_cols + num_cols
    
    ## Filling missing values as a basic cleaning step.
    df[num_cols] = df[num_cols].fillna(0)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return df, X, y, cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor


def make_dataloaders(X, y, preprocessor):
    ## Fitting preprocessor on full X (train+val+test) to have a stable space,
    ## then splittinh into train/val/test on labels.
    X_proc = preprocessor.fit_transform(X)

    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    X_proc = np.asarray(X_proc, dtype=np.float32)

    ## Encoding labels.
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    ## Splitting into train/val/test.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_proc,
        y_enc,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    ## Wrapping them into PyTorch tensors and DataLoaders.
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).long()

    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).long()

    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train.shape[1],
        len(label_encoder.classes_),
        label_encoder,
        preprocessor,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += xb.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def test_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    return test_acc


def main():
    df, X, y, cat_cols, num_cols = load_dataset()
    preprocessor = build_preprocessor(cat_cols, num_cols)

    (
        train_loader,
        val_loader,
        test_loader,
        input_dim,
        num_classes,
        label_encoder,
        preprocessor,
    ) = make_dataloaders(X, y, preprocessor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="mlp_full_run"):

        mlflow.log_param("hidden_dim", HIDDEN_DIM)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("lr", LR)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("num_classes", num_classes)

        model = MLP(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=num_classes)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            print(
                f"Epoch {epoch:02d} | "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "model_state_dict": model.state_dict(),
                }

        ## Loading best model for test.
        if best_state is not None:
            model.load_state_dict(best_state["model_state_dict"])

        test_acc = test_model(model, test_loader, device)
        print(f"Test accuracy: {test_acc:.4f}")
        mlflow.log_metric("test_acc", test_acc)

        ## Saving artifacts (model + preprocessor + label encoder).
        model_path = ARTIFACTS_DIR / "mlp_full.pt"
        torch.save(
            {
                "input_dim": input_dim,
                "hidden_dim": HIDDEN_DIM,
                "output_dim": num_classes,
                "model_state_dict": model.state_dict(),
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path))

        preproc_path = ARTIFACTS_DIR / "preproc_full.pkl"
        with open(preproc_path, "wb") as f:
            pickle.dump(preprocessor, f)
        mlflow.log_artifact(str(preproc_path))

        le_path = ARTIFACTS_DIR / "label_encoder_full.pkl"
        with open(le_path, "wb") as f:
            pickle.dump(label_encoder, f)
        mlflow.log_artifact(str(le_path))

        print(f"Saved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()