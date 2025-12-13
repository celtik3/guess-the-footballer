import os
import pandas as pd
from typing import Optional, Dict, Any
from difflib import get_close_matches


CSV_PATH = os.path.join("data", "processed", "players_full.csv")


def load_players_df() -> pd.DataFrame:
    """Load the processed full players dataset."""
    df = pd.read_csv(CSV_PATH)
    return df


def get_player_by_id(df: pd.DataFrame, player_id: int) -> Optional[pd.Series]:
    """Return a player row by player_id, or None if not found."""
    row = df[df["player_id"] == player_id]
    if row.empty:
        return None
    return row.iloc[0]


## This definitely will be expanded in the future with more sophisticated matching.
def get_player_by_name(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """
    Return a player row by name, with some fuzzy matching:
      1) exact (case-insensitive) match
      2) substring match (e.g. 'icardi' -> 'Mauro Icardi')
      3) closest match by string similarity (e.g. 'Mbape' -> 'Kylian Mbappe')
    """
    if not name:
        return None

    name_norm = name.strip().lower()

    ## Exact (case-insensitive) match
    row = df[df["player_name"].str.lower() == name_norm]
    if not row.empty:
        return row.iloc[0]

    ## Substring match: guess inside full name
    mask_contains = df["player_name"].str.lower().str.contains(name_norm, na=False)
    if mask_contains.any():
        return df[mask_contains].iloc[0]

    ## Fuzzy match using difflib
    all_names = df["player_name"].tolist()
    best = get_close_matches(name, all_names, n=1, cutoff=0.5)
    if best:
        row = df[df["player_name"] == best[0]]
        if not row.empty:
            return row.iloc[0]

    return None


def player_to_features_dict(row: pd.Series) -> Dict[str, Any]:
    """
    Convert a player row into a dict used by the game and similarity.
    Includes player_id, so I can use cached embeddings.
    """
    return {
        "player_id": int(row["player_id"]),
        "player_name": row.get("player_name", ""),

        ## For clue fields.
        "position_group": row.get("position_group", "Other"),
        "age_band": row.get("age_band", "Unknown"),
        "nation_region": row.get("nation_region", "Unknown"),
        "league_region": row.get("league_region", "Unknown"),
        "style": row.get("style", "all-rounder"),
        "peak_year": row.get("peak_year", "Unknown"),
        "value_bin": row.get("value_bin", "Unknown"),
        "current_club_name": row.get("current_club_name", "Unknown Club"),
    }