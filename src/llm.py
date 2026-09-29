"""
llm.py
Phase 10: LLM integration (Gemini API).

Flow:
  NLP report -> fill prompt template -> API call -> parse JSON -> result
If the API fails (no key, network, quota, bad JSON), a rule-based FALLBACK
correction built from the spelling module is returned instead.
"""

import json
import os
import re
import time

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

CONFIG_PATH = "config/config.yaml"
REQUIRED_KEYS = {"corrected_text", "changes", "summary"}

_client = None  # created once and reused


# ---------- Configuration and prompt ----------

def load_llm_config():
    """Read the 'llm' section of config/config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["llm"]


def load_prompt_template(path):
    """Read the prompt text file."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def build_prompt(prompt_fields, template=None):
    """Fill {placeholders} in the template with the NLP results.

    We replace only {word_characters} placeholders in ONE pass, so the JSON
    braces in the prompt are untouched, and text typed by the user can never
    be mistaken for a placeholder."""
    if template is None:
        template = load_prompt_template(load_llm_config()["prompt_file"])
    return re.sub(
        r"\{(\w+)\}",
        lambda match: str(prompt_fields.get(match.group(1), match.group(0))),
        template,
    )


# ---------- API call ----------

def get_client():
    """Create the Gemini client the first time it is needed."""
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found. Create a .env file (see .env.example).")
        _client = genai.Client(api_key=api_key)
    return _client


def call_llm(prompt, config):
    """Send the prompt to Gemini and return the raw response text."""
    response = get_client().models.generate_content(
        model=config["model"],
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=config.get("temperature", 0.2),
            response_mime_type="application/json",  # ask for JSON output
        ),
    )
    return response.text


def parse_llm_json(text):
    """Turn the model's reply into a Python dictionary and validate its shape.
    Raises ValueError if the reply is unusable."""
    if not text:
        raise ValueError("The model returned an empty response.")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())  # remove ``` fences
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("The response is not a JSON object.")
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"The response is missing keys: {sorted(missing)}")
    if not isinstance(data["changes"], list):
        raise ValueError("'changes' must be a list.")
    return data


# ---------- Fallback when the API is unavailable ----------

def fallback_correction(pipeline_result):
    """Rule-based correction: replace each misspelled word by its best
    candidate in context. It cannot fix grammar (that needs the LLM)."""
    text = pipeline_result["normalized_text"]
    changes = []
    for error in pipeline_result["spelling_errors"]:
        best = error["best_in_context"]
        if not best:
            continue
        original = error["word"]
        replacement = best.capitalize() if original[0].isupper() else best
        text, count = re.subn(r"\b" + re.escape(original) + r"\b", replacement, text, count=1)
        if count:
            changes.append({
                "original": original,
                "corrected": replacement,
                "type": "spelling",
                "explanation": "Not in the lexicon; best candidate by edit distance and n-gram context.",
            })
    return text, changes


# ---------- Main entry point ----------

def correct_text(pipeline_result):
    """Correct the text using the NLP report + LLM.

    Returns a dictionary:
      source         "llm" or "fallback"
      corrected_text, changes, summary
      error          None, or the reason the LLM could not be used
      prompt         the exact prompt that was sent (for transparency)"""
    config = load_llm_config()
    prompt = build_prompt(pipeline_result["prompt_fields"])
    retries = config.get("max_retries", 2)
    last_error = None

    for attempt in range(retries + 1):
        try:
            data = parse_llm_json(call_llm(prompt, config))
            return {
                "source": "llm",
                "corrected_text": data["corrected_text"],
                "changes": data["changes"],
                "summary": data["summary"],
                "error": None,
                "prompt": prompt,
            }
        except RuntimeError as error:      # missing key: retrying cannot help
            last_error = error
            break
        except Exception as error:         # network, quota, bad JSON ...
            last_error = error
            if attempt < retries:
                time.sleep(2 ** attempt)   # wait 1s, then 2s, before retrying

    text, changes = fallback_correction(pipeline_result)
    return {
        "source": "fallback",
        "corrected_text": text,
        "changes": changes,
        "summary": "LLM unavailable: only spelling corrections were applied.",
        "error": str(last_error),
        "prompt": prompt,
    }