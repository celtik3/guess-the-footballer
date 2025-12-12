import os
import pickle
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.mlp_model import MLP


ARTIFACTS_DIR = "artifacts"
PREPROC_PATH = os.path.join(ARTIFACTS_DIR, "preproc.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "mlp.pt")


class ModelBundle:
    """
    Holds the loaded preprocessor, label encoder, and PyTorch MLP model.
    Provides convenience methods for encoding features and computing embeddings.
    """

    def __init__(self):
        ## Loading preprocessor.
        with open(PREPROC_PATH, "rb") as f:
            self.preprocessor = pickle.load(f)

        ## Loading label encoder.
        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        ## Loading model state.
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        input_dim = checkpoint["input_dim"]
        hidden_dim = checkpoint["hidden_dim"]
        output_dim = checkpoint["output_dim"]

        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def features_to_tensor(self, features: Dict[str, Any]) -> torch.Tensor:
        """
        Convert a single player's feature dict into a model-ready tensor.
        """
        ## Here, preprocessor expects a 2D array / DataFrame; 
        ## I build a one-row dict as shape (1, n_features_raw), then transform.
        df = pd.DataFrame([features])
        X_processed = self.preprocessor.transform(df)

        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        X_tensor = torch.from_numpy(np.asarray(X_processed)).float()  ## (1, input_dim)
        return X_tensor

    def predict_logits(self, features: Dict[str, Any]) -> torch.Tensor:
        """
        Run the model and return raw logits (1, num_classes).
        """
        x = self.features_to_tensor(features)
        with torch.no_grad():
            logits = self.model(x)
        return logits

    def get_embedding(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Get an embedding vector for a player.
        I use the output of the last hidden layer as the embedding.

        Model architecture:
            Linear -> ReLU -> Linear -> ReLU -> Linear
        
        I take everything except the final Linear layer.
        """
        x = self.features_to_tensor(features)
        with torch.no_grad():
            ## Since self.model.net is a torch.nn.Sequential,I can take all but last layer.
            hidden = self.model.net[:-1](x)  ## (1, hidden_dim)
        ## So, it returns as 1D numpy array.
        return hidden.numpy().reshape(-1)

    def label_index_to_player_id(self, idx: int) -> int:
        """
        Convert a class index (0..num_classes-1) back to the original player_id.
        """
        player_id = int(self.label_encoder.inverse_transform([idx])[0])
        return player_id


## Simple singleton-like helper to avoid reloading every time.
_bundle: ModelBundle = None


def get_model_bundle() -> ModelBundle:
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle