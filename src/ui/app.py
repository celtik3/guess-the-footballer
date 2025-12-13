import gradio as gr
import requests

API_BASE = "http://127.0.0.1:8000"


def _add_message(chat_history, role, content):
    """
    Append a chat message in the Gradio v6 format:
    {"role": "user" | "assistant", "content": "..."}
    """
    chat_history = chat_history or []
    chat_history.append({"role": role, "content": content})
    return chat_history


def start_new_game(chat_history):
    """
    Calls /new_game on the FastAPI backend and updates chat history.
    """
    try:
        resp = requests.post(f"{API_BASE}/new_game")
        resp.raise_for_status()
        data = resp.json()
        msg = f"{data['message']} (Players in pool: {data['num_players']})"
    except Exception as e:
        msg = f"Error starting new game: {e}"

    chat_history = _add_message(chat_history, "assistant", msg)
    ## Returning both updated state and what the Chatbot should display.
    return chat_history, chat_history


def get_clue(chat_history):
    """
    Calls /clue on the FastAPI backend and updates chat history.
    """
    try:
        resp = requests.get(f"{API_BASE}/clue")
        resp.raise_for_status()
        data = resp.json()
        msg = f"Clue: {data['clue']}"
    except Exception as e:
        msg = f"Error getting clue: {e}"

    chat_history = _add_message(chat_history, "assistant", msg)
    return chat_history, chat_history


def make_guess(guess, chat_history):
    """
    Calls /guess on the FastAPI backend with the user's guess.
    """
    guess = (guess or "").strip()
    chat_history = chat_history or []

    if not guess:
        chat_history = _add_message(chat_history, "user", "(empty guess)")
        chat_history = _add_message(
            chat_history,
            "assistant",
            "Please type a player name from the mini dataset.",
        )
        ## Clearing input, update state and Chatbot
        return "", chat_history, chat_history

    ## Adding user's message.
    chat_history = _add_message(chat_history, "user", guess)

    try:
        resp = requests.post(
            f"{API_BASE}/guess",
            json={"guess_name": guess},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        chat_history = _add_message(chat_history, "assistant", f"Error sending guess: {e}")
        return "", chat_history, chat_history

    feedback = data.get("feedback", "No feedback from server.")
    chat_history = _add_message(chat_history, "assistant", feedback)

    if data.get("game_over"):
        chat_history = _add_message(
            chat_history,
            "assistant",
            "Game over! Click 'Start New Game' to play again.",
        )

    ## Clearing the input box after each guess.
    return "", chat_history, chat_history


def build_interface():
    with gr.Blocks(title="Guess the Footballer") as demo:
        gr.Markdown("# ⚽ Guess the Footballer\nA tiny MLOps + DL + FastAPI demo.")

        ## Chat history as list[{"role": ..., "content": ...}].
        chat_history = gr.State([])

        chatbox = gr.Chatbot(
            label="Game",
            height=400,
        )

        with gr.Row():
            with gr.Column(scale=1):
                new_game_btn = gr.Button("Start New Game")
                guess_input = gr.Textbox(
                    label="Your guess",
                    placeholder="e.g. Mauro Icardi",
                )

            with gr.Column(scale=1):
                clue_btn = gr.Button("Get Clue")
                guess_btn = gr.Button("Submit Guess")

        ## Wire buttons
        new_game_btn.click(
            fn=start_new_game,
            inputs=chat_history,
            outputs=[chat_history, chatbox],
        )
        
        clue_btn.click(
            fn=get_clue,
            inputs=chat_history,
            outputs=[chat_history, chatbox],
        )

        guess_btn.click(
            fn=make_guess,
            inputs=[guess_input, chat_history],
            outputs=[guess_input, chat_history, chatbox],
        )

        return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()