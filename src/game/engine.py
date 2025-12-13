from typing import Dict, Any

from src.game.dataset import (
    load_players_df,
    get_player_by_name,
    get_player_by_id,
    player_to_features_dict,
)
from src.game.similarity import player_similarity

from src.llm.host import (
    generate_clue,
    generate_guess_feedback,
    generate_unknown_guess_feedback,
)


class GameEngine:
    """
    Simple in-memory game engine for 'Guess the Footballer'.

    This version:
      - Uses the full processed dataset
      - Picks a random target player
      - Accepts guesses by name
      - Returns similarity and a simple text hint
    """

    def __init__(self):
        self.df = load_players_df()
        self.target_player_id = None
        ## How many clues have been given in the current game.
        self.clue_step = 0  

    def new_game(self) -> Dict[str, Any]:
        row = self.df.sample(1).iloc[0]
        self.target_player_id = int(row["player_id"])
        self.clue_step = 0

        return {
            "message": "New game started! I picked a player from the dataset.",
            "num_players": len(self.df),
        }

    def _get_target_row(self):
        if self.target_player_id is None:
            raise ValueError("No active game. Call new_game() first.")
        row = get_player_by_id(self.df, self.target_player_id)
        if row is None:
            raise ValueError(f"Target player_id {self.target_player_id} not found in df.")
        return row

    def guess(self, guess_name: str) -> Dict[str, Any]:
        """
        User guesses a player by name.
        Returns dict with:
          - correct: bool
          - game_over: bool
          - guessed_name
          - target_same_league
          - target_same_position
          - similarity metrics
          - feedback: host-style text feedback
        """
        if self.target_player_id is None:
            return {
                "error": "No active game. Call new_game() first."
            }

        target_row = self._get_target_row()

        guessed_row = get_player_by_name(self.df, guess_name)
        if guessed_row is None:
            ## Using host stub for unknown name feedback.
            feedback = generate_unknown_guess_feedback(guess_name)
            return {
                "correct": False,
                "game_over": False,
                "guessed_name": guess_name,
                "feedback": feedback,
            }

        guessed_id = int(guessed_row["player_id"])
        correct = (guessed_id == self.target_player_id)

        ## Basic feature comparison for warm/cold hints.
        target_feats = player_to_features_dict(target_row)
        guessed_feats = player_to_features_dict(guessed_row)

        sim = player_similarity(target_feats, guessed_feats)

        same_league = target_row["league_region"] == guessed_row["league_region"]
        same_position = target_row["position_group"] == guessed_row["position_group"]

        ## Simple handcrafted feedback for now in the simple game version.
        if correct:
            game_over = True
            ## Resetting the target so user must start a new game.
            self.target_player_id = None
        else:
            game_over = False

        ## Building context for the host stub
        feedback_context = {
            "correct": correct,
            "guessed_name": guessed_row["player_name"],
            "same_position": same_position,
            "same_league": same_league,
            "cosine_sim": sim.get("cosine_sim"),
            "clue_step": self.clue_step,
        }
        feedback = generate_guess_feedback(feedback_context)

        return {
            "correct": correct,
            "game_over": game_over,
            "guessed_name": guessed_row["player_name"],
            "target_player_id": self.target_player_id if correct else None,
            ## The game won't reveal the name unless correct.
            "same_league_region": same_league,
            "same_position_group": same_position,
            "similarity": sim,
            "feedback": feedback,
        }
    

    def get_basic_clue(self):
        """
        Return a progressive clue about the current target player using the LLM-like host stub.

        Each time this method is called during a game, I advance the clue_step:
          step 1: very general (position)
          step 2: add league region
          step 3: add naton region
          step 4: add nationality
          step 5: add style
          step 6: add peak year + value
          step 7: add club name
        """
        if self.target_player_id is None:
            return {"error": "No active game. Call new_game() first."}

        target_row = self._get_target_row()

        ## Building structured facts dict for the host.
        facts = {
            "position_group": target_row["position_group"],
            "league_region": target_row["league_region"],
            "nation_region": target_row["nation_region"],
            "nationality": target_row.get("nationality", "Unknown"),
            "style": target_row["style"],
            "peak_year": int(target_row["peak_year"]),
            "value_bin": target_row["value_bin"],
            "current_club_name": target_row.get("current_club_name", "Unknown Club"),
        }

        ## Advance clue step for this game.
        self.clue_step += 1
        clue_text = generate_clue(facts, step=self.clue_step)

        return {"clue": clue_text, "clue_step": self.clue_step}
    