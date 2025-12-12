import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "player-scores"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


## Basic mapping functions/helpers.

def map_position_group(pos: Optional[str]) -> str:
    """
    Map the broad 'position' column from players.csv:
    Attack, Midfield, Defender, Goalkeeper, Missing
    into the normalized position group labels.
    """
    if not isinstance(pos, str):
        return "Other"
    p = pos.strip().lower()
    if p == "goalkeeper":
        return "GK"
    if p == "defender":
        return "Defender"
    if p == "midfield":
        return "Midfielder"
    if p == "attack":
        return "Forward"
    return "Other"


## very coarse country --> region mapping, just enough for features / clues for now.
EUROPE = {
    "England", "Scotland", "Wales", "Northern Ireland", "Ireland",
    "Spain", "France", "Germany", "Italy", "Portugal", "Netherlands",
    "Belgium", "Switzerland", "Austria", "Croatia", "Serbia", "Bosnia-Herzegovina",
    "Slovenia", "Slovakia", "Czech Republic", "Poland", "Hungary", "Denmark",
    "Sweden", "Norway", "Finland", "Iceland", "Greece", "Turkey", "Ukraine",
    "Russia", "Romania", "Bulgaria"
}
SOUTH_AMERICA = {
    "Brazil", "Argentina", "Uruguay", "Chile", "Colombia", "Peru",
    "Paraguay", "Ecuador", "Venezuela", "Bolivia"
}
NORTH_AMERICA = {
    "USA", "United States", "Canada", "Mexico", "Costa Rica",
    "Honduras", "Panama", "Jamaica"
}
AFRICA = {
    "Nigeria", "Ghana", "Ivory Coast", "Côte d'Ivoire", "Senegal",
    "Cameroon", "Algeria", "Morocco", "Egypt", "Tunisia", "Mali",
}
ASIA = {
    "Japan", "South Korea", "Korea, South", "China", "Saudi Arabia",
    "Qatar", "Iran", "Iraq", "Australia"
}


def map_country_to_region(country: Optional[str]) -> str:
    if not isinstance(country, str) or not country:
        return "Other"
    c = country.strip()
    if c in EUROPE:
        return "Europe"
    if c in SOUTH_AMERICA:
        return "South America"
    if c in NORTH_AMERICA:
        return "North America"
    if c in AFRICA:
        return "Africa"
    if c in ASIA:
        return "Asia"
    return "Other"


def age_from_birthdate(date_str: str, ref_year: int = 2025) -> Optional[int]:
    if not isinstance(date_str, str):
        return None
    try:
        dt = pd.to_datetime(date_str, errors="coerce")
    except Exception:
        return None
    if pd.isna(dt):
        return None
    return int(ref_year - dt.year)


def age_band(age: Optional[int]) -> str:
    if age is None or math.isnan(age):
        return "Unknown"
    if age < 20:
        return "<20"
    if age <= 24:
        return "20-24"
    if age <= 28:
        return "24-28"
    if age <= 31:
        return "28-31"
    if age <= 35:
        return "31-35"
    return "36+"


def value_bin_from_eur(v: float) -> str:
    if pd.isna(v) or v <= 5_000_000:
        return "Low"
    if v <= 20_000_000:
        return "Medium"
    if v <= 60_000_000:
        return "High"
    return "Elite"


def style_from_stats(pos_group: str, goals_per90: float, assists_per90: float) -> str:
    if pos_group == "Forward":
        if goals_per90 >= 0.6:
            return "goal machine"
        if assists_per90 >= 0.35:
            return "playmaker"
        return "attacking threat"
    if pos_group == "Midfielder":
        if assists_per90 >= 0.3:
            return "playmaker"
        return "all-round midfielder"
    if pos_group == "Defender":
        return "defensive wall"
    if pos_group == "GK":
        return "shot stopper"
    return "all-rounder"


## Core feature building functions.

def load_raw_tables():
    players = pd.read_csv(RAW_DIR / "players.csv")
    appearances = pd.read_csv(RAW_DIR / "appearances.csv")
    game_events = pd.read_csv(RAW_DIR / "game_events.csv")
    player_vals = pd.read_csv(RAW_DIR / "player_valuations.csv")
    clubs = pd.read_csv(RAW_DIR / "clubs.csv")
    games = pd.read_csv(RAW_DIR / "games.csv")
    competitions = pd.read_csv(RAW_DIR / "competitions.csv")
    return players, appearances, game_events, player_vals, clubs, games, competitions


def build_appearance_aggregates(appearances: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate minutes, games, heuristic 'starts', goals, assists per player
    using appearances.csv schema:

    appearance_id, game_id, player_id, ..., goals, assists, minutes_played
    """
    df = appearances.copy()

    ## Grouping by player and aggregating.
    agg = (
        df.groupby("player_id")
        .agg(
            total_minutes=("minutes_played", "sum"),
            games_played=("game_id", "nunique"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
        )
        .reset_index()
    )

    ## Here, I assume the player started the game if played at least 60 mins.
    ## This is a heuristic to make it easier without having to join full lineups.
    df["is_start_like"] = df["minutes_played"] >= 60
    starts = df.groupby("player_id")["is_start_like"].sum().rename("starts")
    agg = agg.merge(starts, on="player_id", how="left")

    ## Per-90 rates.
    agg["goals_per90"] = np.where(
        agg["total_minutes"] > 0, agg["goals"] * 90.0 / agg["total_minutes"], 0.0
    )
    agg["assists_per90"] = np.where(
        agg["total_minutes"] > 0, agg["assists"] * 90.0 / agg["total_minutes"], 0.0
    )

    ## Starts percentage.
    agg["starts_pct"] = np.where(
        agg["games_played"] > 0, agg["starts"] / agg["games_played"], 0.0
    )

    return agg


def build_latest_and_peak_valuations(player_vals: pd.DataFrame) -> pd.DataFrame:
    df = player_vals.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    ## Latest valuation per player.
    idx_latest = df.groupby("player_id")["date"].idxmax()
    latest = (
        df.loc[idx_latest, ["player_id", "market_value_in_eur", "date"]]
        .rename(columns={"market_value_in_eur": "latest_value_eur", "date": "latest_value_date"})
    )

    ## Peak valuation and year
    idx_peak = df.groupby("player_id")["market_value_in_eur"].idxmax()
    peak = (
        df.loc[idx_peak, ["player_id", "market_value_in_eur", "date"]]
        .rename(columns={"market_value_in_eur": "peak_value_eur", "date": "peak_value_date"})
    )
    peak["peak_year"] = peak["peak_value_date"].dt.year

    out = latest.merge(peak[["player_id", "peak_value_eur", "peak_year"]], on="player_id", how="left")
    return out


def build_league_region(players: pd.DataFrame, clubs: pd.DataFrame, games: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    """
    Coarse league_region from current club's domestic competition country/confederation.
    """
    ## Competitions --> expect something like 'country_name' or 'sub_type' / 'type'
    comp_cols = competitions.columns
    country_col = "country_name" if "country_name" in comp_cols else None
    if country_col is None and "country_name" in clubs.columns:
        ## Fallback, so I use club country.
        country_col = "country_name"

    clubs2 = clubs.copy()
    comps2 = competitions.copy()

    ## Mapping domestic_competition_id to competition info.
    if "domestic_competition_id" in clubs2.columns and "competition_id" in comps2.columns:
        clubs2 = clubs2.merge(
            comps2[["competition_id", country_col]] if country_col else comps2[["competition_id"]],
            left_on="domestic_competition_id",
            right_on="competition_id",
            how="left",
            suffixes=("", "_comp"),
        )

    ## Here, I join players and clubs to get league region.
    if "current_club_id" in players.columns:
        merged = players.merge(
            clubs2[["club_id", country_col]] if country_col else clubs2[["club_id"]],
            left_on="current_club_id",
            right_on="club_id",
            how="left",
        )
    else:
        merged = players.copy()
        merged[country_col] = np.nan if country_col else np.nan

    if country_col:
        merged["league_region"] = merged[country_col].apply(map_country_to_region)
    else:
        merged["league_region"] = "Unknown"

    return merged[["player_id", "league_region"]]


def build_players_full():
    players, appearances, game_events, player_vals, clubs, games, competitions = load_raw_tables()

    df = players.copy()
    df = df.rename(columns={"name": "player_name"}) if "name" in df.columns else df

    ## Adding current club information.
    if "current_club_name" in df.columns:
        df["current_club_name"] = df["current_club_name"].fillna("Unknown Club")
    else:
        df["current_club_name"] = "Unknown Club"

    if "current_club_domestic_competition_id" in df.columns:
        df["current_club_domestic_competition_id"] = (
            df["current_club_domestic_competition_id"].fillna("Unknown")
        )
    else:
        df["current_club_domestic_competition_id"] = "Unknown"
    
    if "date_of_birth" in df.columns:
        df["age"] = df["date_of_birth"].apply(age_from_birthdate)
    else:
        df["age"] = np.nan
    df["age_band"] = df["age"].apply(age_band)

    country_col = "country_of_citizenship" if "country_of_citizenship" in df.columns else (
        "country_name" if "country_name" in df.columns else None
    )
    if country_col:
        df["nation_region"] = df[country_col].apply(map_country_to_region)
    else:
        df["nation_region"] = "Other"

    ## Position group.
    if "position" in df.columns:
        df["position_group"] = df["position"].apply(map_position_group)
    else:
        df["position_group"] = "Other"

    ## League region from clubs and competitions.
    leagues = build_league_region(df, clubs, games, competitions)
    df = df.merge(leagues, on="player_id", how="left")

    ## Aappearances-based stats.
    app_agg = build_appearance_aggregates(appearances, games)
    df = df.merge(app_agg, on="player_id", how="left")

    ## Valuation stats.
    val_agg = build_latest_and_peak_valuations(player_vals)
    df = df.merge(val_agg, on="player_id", how="left")

    ## Style.
    df["style"] = df.apply(
        lambda row: style_from_stats(
            row["position_group"],
            float(row.get("goals_per90", 0.0) or 0.0),
            float(row.get("assists_per90", 0.0) or 0.0),
        ),
        axis=1,
    )

    ## Value bin.
    df["value_bin"] = df["peak_value_eur"].apply(value_bin_from_eur)

    ## Basic cleaning and filtering.
    df["total_minutes"] = df["total_minutes"].fillna(0)
    df["games_played"] = df["games_played"].fillna(0)
    df["goals"] = df["goals"].fillna(0)
    df["assists"] = df["assists"].fillna(0)

    ## I keep only players with some meaningful minutes and a valuation.
    df = df[df["total_minutes"] >= 900]
    df = df[~df["peak_value_eur"].isna()]

    ## And, I sort them by peak value and keep a manageable subset.
    df = df.sort_values("peak_value_eur", ascending=False)
    df_top = df.head(500).reset_index(drop=True)

    ## These are the columns I care and keep for the final output.
    cols = [
        "player_id",
        "player_name",
        "age",
        "age_band",
        "position_group",
        "nation_region",
        "league_region",
        "current_club_name",
        "current_club_domestic_competition_id",
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
        "value_bin",
        "style",
    ]
    df_top = df_top[cols]

    out_path = PROC_DIR / "players_full.csv"
    df_top.to_csv(out_path, index=False)
    print(f"Saved {len(df_top)} players to {out_path}")


if __name__ == "__main__":
    build_players_full()