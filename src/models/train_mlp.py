import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import mlflow
import mlflow.pytorch

from src.models.mlp_model import MLP


## Paths that I will use to train and save the model and artifacts.
X_PATH = os.path.join("data", "processed", "X.npy")
Y_PATH = os.path.join("data", "processed", "y.npy")
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "mlp.pt")
PREPROC_PATH = os.path.join(ARTIFACTS_DIR, "preproc.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")


def load_data():
    print(f"Loading features from {X_PATH}")
    X = np.load(X_PATH)
    print(f"Loading labels from {Y_PATH}")
    y = np.load(Y_PATH)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    ## Converting to torch tensors.
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset, X.shape[1], len(np.unique(y))


def train_mlp(
    hidden_dim=64,
    batch_size=4,
    lr=1e-2,
    num_epochs=100,
):
    ## Ensuring the existance of artifacts dir.
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    ## Loading data
    dataset, input_dim, num_classes = load_data()

    ## Since this is a miniset,I can shuffle and use a small batch size.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ## I don’t have a GPU on my machine.
    device = torch.device("cpu")
    print(f"Using device: {device}")

    ## Model creation.
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## Setting MLflow experiment name (This is what I’ll see this in the UI later.)
    mlflow.set_experiment("guess_baller_mlp")

    with mlflow.start_run():
        ## Logging the hyperparameters.
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("num_classes", num_classes)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X_batch.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Loss --> {epoch_loss:.4f} | Acc --> {epoch_acc:.4f}"
            )

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", epoch_loss, step=epoch + 1)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch + 1)

        # Save model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": num_classes,
            },
            MODEL_PATH,
        )
        print(f"Saved trained model to {MODEL_PATH}")

        ## Logging model and preprocessing artifacts to MLflow.
        mlflow.log_artifact(MODEL_PATH)

        if os.path.exists(PREPROC_PATH):
            mlflow.log_artifact(PREPROC_PATH)
        if os.path.exists(LABEL_ENCODER_PATH):
            mlflow.log_artifact(LABEL_ENCODER_PATH)

        mlflow.pytorch.log_model(model, name="model")


if __name__ == "__main__":
    train_mlp()