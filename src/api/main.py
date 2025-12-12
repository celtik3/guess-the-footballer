from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.game.engine import GameEngine
from src.game.inference import get_model_bundle

app = FastAPI(
    title="Guess the Footballer API",
    version="0.1.0",
    description="Simple API for the Guess the Footballer game (mini dataset).",
)

## Single global engine instance for now, so there will beone active game at a time.
engine = GameEngine()

## Warming up model/preprocessor on startup so first guess is faster.
_ = get_model_bundle()


## Pydantic models for request/response bodies.

class NewGameResponse(BaseModel):
    message: str
    num_players: int


class GuessRequest(BaseModel):
    guess_name: str


class GuessResponse(BaseModel):
    correct: bool
    game_over: bool = False
    guessed_name: Optional[str] = None
    feedback: str
    same_league_region: Optional[bool] = None
    same_position_group: Optional[bool] = None
    cosine_sim: Optional[float] = None
    euclidean_dist: Optional[float] = None


class ClueResponse(BaseModel):
    clue: str


## Route handlers 

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/new_game", response_model=NewGameResponse)
def new_game():
    """
    Start a new game with a random target player.
    """
    info = engine.new_game()
    return NewGameResponse(
        message=info["message"],
        num_players=info["num_players"],
    )


@app.post("/guess", response_model=GuessResponse)
def make_guess(payload: GuessRequest):
    """
    Make a guess by player name.
    """
    result = engine.guess(payload.guess_name)

    ## If engine reports an error, such as unknown name or no active game.
    if "error" in result:
        return GuessResponse(
            correct=False,
            guessed_name=None,
            feedback=result["error"],
        )

    sim = result.get("similarity", {})

    return GuessResponse(
        correct=result["correct"],
        game_over=result.get("game_over", False),
        guessed_name=result.get("guessed_name"),
        feedback=result.get("feedback", ""),
        same_league_region=result.get("same_league_region"),
        same_position_group=result.get("same_position_group"),
        cosine_sim=sim.get("cosine_sim"),
        euclidean_dist=sim.get("euclidean_dist"),
    )


@app.get("/clue", response_model=ClueResponse)
def get_clue():
    """
    Get a simple clue for the current target player.
    """
    info = engine.get_basic_clue()
    if "error" in info:
        ## FastAPI will still return HTTP 200 here with a simple message;
        ## I can improve error handling later if needed.
        return ClueResponse(clue=info["error"])

    return ClueResponse(clue=info["clue"])