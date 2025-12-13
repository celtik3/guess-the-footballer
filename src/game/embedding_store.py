from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ART_DIR = PROJECT_ROOT / "artifacts_full"

EMB_PATH = ART_DIR / "embeddings.npy"
IDX_PATH = ART_DIR / "player_index.csv"

@dataclass
class EmbeddingStore:
    embeddings: np.ndarray ## (N, D)
    id_to_row: dict ## player_id --> row_index

    @classmethod
    def load(cls) -> "EmbeddingStore":
        emb = np.load(EMB_PATH)
        idx = pd.read_csv(IDX_PATH)
        id_to_row = dict(zip(idx["player_id"].astype(int), idx["row_index"].astype(int)))
        return cls(embeddings=emb, id_to_row=id_to_row)

    def vec(self, player_id: int) -> np.ndarray:
        return self.embeddings[self.id_to_row[int(player_id)]]
