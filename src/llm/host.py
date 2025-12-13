import random
from typing import Dict, Any, Optional

## Optional HF integration.
try:
    from .hf_host import generate_clue_hf
    HF_AVAILABLE = True
except ImportError:
    generate_clue_hf = None  # type: ignore
    HF_AVAILABLE = False

## When I toggle this to True, I actually call the HF model.
USE_HF_FOR_CLUES = False


def generate_clue(facts: Dict[str, Any], step: int) -> str:
    """
    Entry point for clue generation used by the game engine.

    If USE_HF_FOR_CLUES is True and HF is available, use the Hugging Face model.
    Otherwise, fall back to the lightweight stub with progressive templates.
    """
    if USE_HF_FOR_CLUES and HF_AVAILABLE and generate_clue_hf is not None:
        try:
            return generate_clue_hf(facts, step)
        except Exception:
            ## On any HF failure, it falls back to stub so the game still works.
            pass

    ## Stub implementation with progressive clues
    position = str(facts.get("position_group", "player"))
    league_region = facts.get("league_region", "Europe")
    nation_region = facts.get("nation_region", "Europe")
    nationality = facts.get("nationality", "Unknown")
    style = str(facts.get("style", "all-rounder"))
    peak_year = facts.get("peak_year", "recent years")
    value_bin = facts.get("value_bin", "High")
    club = facts.get("current_club_name", "Unknown Club")

    ## Clamp step so I don't go beyond our designed levels.
    step = max(1, min(step, 7))

    ## Level 1 --> very general, only position.
    if step == 1:
        templates = [
            f"The player is a {position}.",
            f"You're looking for a {position}.",
        ]

    ## Level 2 --> revealing league region.
    elif step == 2:
        templates = [
            f"He plays his football in {league_region}.",
            f"This {position} is active in {league_region} leagues.",
        ]

    ## Level 3 --> adding nation region.
    elif step == 3:
        templates = [
            f"He comes from the {nation_region} region.",
            f"The player originates from {nation_region}.",
        ]

    # Level 4 --> adding nationality.
    elif step == 4:
        templates = [
            f"Well, his nationality is {nationality}.",
        ]

    ## Level 5 --> adding his style.
    elif step == 5:
        templates = [
            f"On the pitch, he is known as a {style}.",
            f"His playing style can be described as {style}.",
        ]

    ## Level 6 --> peak year and value band
    elif step == 6:
        templates = [
            f"He reached his peak around {peak_year} and sits in the {value_bin} market value tier.",
            f"His market value peaked near {peak_year}, placing him in the {value_bin} range.",
        ]

    ## Level 7 --> last clue as current club.
    else:
        templates = [
            f"Final clue, he currently plays for {club}.",
            f"Last hint is his club, he plays for {club}.",
        ]

    template = random.choice(templates)
    clue = template.format(
        position=position,
        league_region=league_region,
        nation_region=nation_region,
        style=style,
        peak_year=peak_year,
        value_bin=value_bin,
        club=club,
    )

    return clue


def generate_unknown_guess_feedback(raw_name: str) -> str:
    """
    Feedback when the system can't map the user's input to any player.
    """
    cleaned = raw_name.strip() or "that"
    templates = [
        f"I’m not sure who you mean by '{cleaned}'. Try a well-known player from the current pool.",
        f"I couldn't match '{cleaned}' to anyone in this mini dataset. Maybe check the spelling?",
        f"'{cleaned}' doesn't ring a bell for this game. Try another famous player.",
    ]
    return random.choice(templates)


def generate_guess_feedback(context: Dict[str, Any]) -> str:
    """
    LLM-like host stub for guess feedback.

    Expects context dict with keys:
      - correct: bool
      - guessed_name: str
      - same_position: bool
      - same_league: bool
      - cosine_sim: float (may be None)
      - clue_step: int (how many clues user has asked for so far)
    """

    correct = bool(context.get("correct", False))
    name = context.get("guessed_name", "that player")
    same_pos = bool(context.get("same_position", False))
    same_league = bool(context.get("same_league", False))
    cos_sim = context.get("cosine_sim", None)
    clue_step = int(context.get("clue_step", 0))

    ## If correct, being enthusiastic and closing the loop.
    if correct:
        templates = [
            f"Spot on! It was indeed {name}. Great job.",
            f"Correct! {name} was the player I had in mind.",
            f"Nice one! You guessed {name} correctly.",
        ]
        return random.choice(templates)

    ## Not correct, so building some hint flavor.
    ## Interpreting cosine similarity (if available) into a coarse 'warmth'.
    warmth = ""
    if isinstance(cos_sim, (int, float)):
        if cos_sim >= 0.9:
            warmth = "You’re extremely close in profile."
        elif cos_sim >= 0.75:
            warmth = "You’re getting quite warm."
        elif cos_sim >= 0.5:
            warmth = "Not bad, but there’s still some distance."
        else:
            warmth = "Overall profile-wise, you’re still quite far from the target."

    overlap_bits = []
    if same_pos:
        overlap_bits.append("same position group")
    if same_league:
        overlap_bits.append("same league region")

    if overlap_bits:
        overlap_text = " and ".join(overlap_bits)
        overlap_sentence = f"Your guess shares the {overlap_text} with the target."
    else:
        overlap_sentence = "Your guess differs in both position and league region."

    ## Slightly different tone depending on how many clues have been requested.
    if clue_step <= 1:
        tone = "early in the game"
    elif clue_step == 2:
        tone = "after a couple of hints"
    else:
        tone = "with several clues already revealed"

    ## Combining everything into one friendly sentence.
    parts = [
        f"Not {name}.",
        overlap_sentence,
        warmth,
        f"We're still {tone}. Try another player!"
    ]
    ## Filtering out empty parts and joining.
    msg = " ".join(p for p in parts if p)
    return msg