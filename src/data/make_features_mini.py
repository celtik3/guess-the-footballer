import os
import numpy as np
import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


## Paths relative to project root.
RAW_CSV_PATH = os.path.join("data", "raw", "players_mini.csv")
PROCESSED_DIR = os.path.join("data", "processed")
ARTIFACTS_DIR = "artifacts"

X_PATH = os.path.join(PROCESSED_DIR, "X.npy")
Y_PATH = os.path.join(PROCESSED_DIR, "y.npy")
PREPROC_PATH = os.path.join(ARTIFACTS_DIR, "preproc.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")


def main():
    ## Ensuring the output folders exist.
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    ## Loading the raw CSV.
    print(f"Loading data from {RAW_CSV_PATH} ...")
    df = pd.read_csv(RAW_CSV_PATH)

    ## Showing cols just to be explicit.
    print("Columns -->", df.columns.tolist())

    ## Defining input features and target.
    ## I will use player_id as the target (class), but encoded to 0..n_classes-1. 
    target_col = "player_id"

    ## Feature columns (everything except player_id and name).
    feature_cols = [
        "position_group",
        "age_band",
        "nation_region",
        "league_region",
        "current_club",
        "goals_per90",
        "assists_per90",
        "minutes_per_season",
        "peak_year",
        "value_bin",
        "style",
    ]

    X_raw = df[feature_cols]
    y_raw = df[target_col]

    ## Defining which columns are categorical vs numeric.
    categorical_features = [
        "position_group",
        "age_band",
        "nation_region",
        "league_region",
        "current_club",
        "value_bin",
        "style",
    ]

    numeric_features = [
        "goals_per90",
        "assists_per90",
        "minutes_per_season",
        "peak_year",
    ]

    ## Building preprocessing pipeline.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    ## Fitting the preprocessor and transform X.
    print("Fitting preprocessing pipeline...")
    X_processed = preprocessor.fit_transform(X_raw)

    ## Converting to dense np array (for simplicity; I might keep sparse in the future for more players)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    ## Encoding labels (player_id --> 0..n_classes-1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    ## Saving outputs to the determined dir.
    print(f"Saving X to {X_PATH}")
    np.save(X_PATH, X_processed)

    print(f"Saving y to {Y_PATH}")
    np.save(Y_PATH, y_encoded)

    print(f"Saving preprocessor to {PREPROC_PATH}")
    with open(PREPROC_PATH, "wb") as f:
        pickle.dump(preprocessor, f)

    print(f"Saving label encoder to {LABEL_ENCODER_PATH}")
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print("Done. Shapes of processed data:")
    print("   X -->", X_processed.shape)
    print("   y -->", y_encoded.shape)

    print("\nClasses mapping (label idx --> player_id):")
    for idx, player_id in enumerate(label_encoder.classes_):
        print(f"  {idx} --> {player_id}")


if __name__ == "__main__":
    main()