from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


## I chose this microsoft phi model, but any small instruct model can be used.

HF_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct" 

_device: Optional[torch.device] = None
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None


def _load_model():
    """
    Lazy-load the HF model and tokenizer on first use.
    Runs on CPU by default.
    """
    global _device, _tokenizer, _model

    if _model is not None:
        return

    _device = torch.device("cpu")
    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
    _model.to(_device)
    _model.eval()


def generate_clue_hf(facts: Dict[str, Any], step: int) -> str:
    """
    Use a Hugging Face instruct model to generate a clue.

    facts: same dict as for the stub:
      - position_group
      - league_region
      - nation_region
      - style
      - peak_year
      - value_bin

    step: 1..4 for progressive hints (general -> specific).
    """

    _load_model()

    position = str(facts.get("position_group", "player")).lower()
    league_region = facts.get("league_region", "Europe")
    nation_region = facts.get("nation_region", "Europe")
    style = str(facts.get("style", "all-rounder")).lower()
    peak_year = facts.get("peak_year", "recent years")
    value_bin = facts.get("value_bin", "High")

    ## Building a short "facts" string.
    facts_text = (
        f"position={position}, league_region={league_region}, "
        f"nation_region={nation_region}, style={style}, "
        f"peak_year={peak_year}, value_bin={value_bin}"
    )

    ## Describing clue granularity.
    if step <= 1:
        level_text = "Give only very general information like position and league region."
    elif step == 2:
        level_text = "You can add the origin region but keep it somewhat general."
    elif step == 3:
        level_text = "You may add style/role but avoid extremely precise details."
    else:
        level_text = "You may give more specific hints, including peak year and value band, but never the name."

    ## Simple instruct-style prompt.
    prompt = (
        "You are the host of a football guessing game.\n"
        "A hidden real player is chosen based on some statistics and metadata.\n"
        "Your job is to give ONE short hint (max ~25 words) to the user.\n"
        "Never reveal or directly name the player.\n"
        f"{level_text}\n"
        f"Facts about the hidden player: {facts_text}\n"
        "Hint:"
    )

    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    ## Decoding only the newly generated part.
    full_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    ## A very rough way to get the hint, so taking text after the last 'Hint:' occurrence.
    if "Hint:" in full_text:
        hint = full_text.split("Hint:")[-1].strip()
    else:
        hint = full_text.strip()

    ## Short post-processing, so collapsing newlines and keeping it simple.
    hint = " ".join(hint.split())
    return hint