"""
test_llm_module.py
Automated tests for src/llm.py. NO real API calls are made: the API function
is replaced by fakes, and the cache is replaced by an in-memory dictionary.
Run from the project root:  python -m pytest -q
"""

import json

import pytest

from src import llm


# ---------- Helpers ----------

def fake_pipeline_result():
    """A minimal NLP report, shaped like the output of run_pipeline()."""
    return {
        "original_text": "I recieved it.",
        "normalized_text": "I recieved it.",
        "spelling_errors": [{"word": "recieved", "best_in_context": "received"}],
        "prompt_fields": {"original_text": "I recieved it.",
                          "normalized_text": "I recieved it."},
    }


@pytest.fixture
def isolated(monkeypatch):
    """No real cache file, no waiting between retries."""
    saved = {}
    monkeypatch.setattr(llm, "load_cache", lambda: {})
    monkeypatch.setattr(llm, "save_to_cache", lambda key, data: saved.update({key: data}))
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    return saved


GOOD_REPLY = json.dumps({
    "corrected_text": "I received it.",
    "changes": [{"original": "recieved", "corrected": "received",
                 "type": "spelling", "explanation": "Not in the lexicon."}],
    "summary": "Fixed one spelling error.",
})


# ---------- JSON validation ----------

def test_parse_accepts_valid_json():
    assert llm.parse_llm_json(GOOD_REPLY)["corrected_text"] == "I received it."


def test_parse_strips_markdown_fences():
    fenced = "```json\n" + GOOD_REPLY + "\n```"
    assert llm.parse_llm_json(fenced)["summary"] == "Fixed one spelling error."


@pytest.mark.parametrize("bad_reply", [
    "",                                   # empty
    "not json at all",                    # not JSON
    '["a", "b"]',                         # JSON, but not an object
    '{"corrected_text": "x"}',            # missing keys
    '{"corrected_text": "x", "changes": "no", "summary": "s"}',  # wrong type
])
def test_parse_rejects_unusable_replies(bad_reply):
    with pytest.raises(ValueError):       # json errors are ValueErrors too
        llm.parse_llm_json(bad_reply)


# ---------- Prompt building ----------

def test_prompt_fills_placeholders_and_keeps_json_braces():
    template = 'Text: {original_text}\nFormat: {"a": 1}'
    prompt = llm.build_prompt({"original_text": "hello"}, template)
    assert prompt == 'Text: hello\nFormat: {"a": 1}'


def test_user_text_cannot_inject_placeholders():
    template = "A: {original_text} B: {spelling_errors}"
    fields = {"original_text": "{spelling_errors}", "spelling_errors": "REAL"}
    assert llm.build_prompt(fields, template) == "A: {spelling_errors} B: REAL"


def test_real_prompt_file_has_all_placeholders_filled():
    fields = {name: "X" for name in [
        "original_text", "normalized_text", "spelling_errors",
        "pos_tags", "ngram_analysis", "dependency_analysis"]}
    assert "{original_text}" not in llm.build_prompt(fields)
    assert "{dependency_analysis}" not in llm.build_prompt(fields)


# ---------- Quota detection ----------

def test_daily_quota_error_is_recognised():
    error = Exception("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")
    assert llm.is_daily_quota_error(error) is True


def test_temporary_error_is_not_a_daily_quota_error():
    assert llm.is_daily_quota_error(Exception("503 UNAVAILABLE high demand")) is False


# ---------- Fallback ----------

def test_fallback_applies_best_spelling_candidate():
    text, changes = llm.fallback_correction(fake_pipeline_result())
    assert text == "I received it."
    assert changes[0]["original"] == "recieved"


def test_fallback_keeps_capital_letter():
    result = {"normalized_text": "Teh cat sat.",
              "spelling_errors": [{"word": "Teh", "best_in_context": "the"}]}
    text, _ = llm.fallback_correction(result)
    assert text == "The cat sat."


# ---------- correct_text: success, failure, retries ----------

def test_successful_llm_answer_is_used_and_saved(monkeypatch, isolated):
    monkeypatch.setattr(llm, "call_llm", lambda prompt, config: GOOD_REPLY)
    result = llm.correct_text(fake_pipeline_result())
    assert result["source"] == "llm"
    assert result["corrected_text"] == "I received it."
    assert len(isolated) == 1             # the answer went into the cache


def test_temporary_failure_retries_then_falls_back(monkeypatch, isolated):
    calls = []

    def always_busy(prompt, config):
        calls.append(1)
        raise Exception("503 UNAVAILABLE high demand")

    monkeypatch.setattr(llm, "call_llm", always_busy)
    result = llm.correct_text(fake_pipeline_result())
    assert result["source"] == "fallback"
    assert result["corrected_text"] == "I received it."     # spelling fixed by fallback
    assert len(calls) == llm.load_llm_config()["max_retries"] + 1
    assert isolated == {}                 # fallback answers are never cached


def test_daily_quota_error_is_not_retried(monkeypatch, isolated):
    calls = []

    def quota_gone(prompt, config):
        calls.append(1)
        raise Exception("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")

    monkeypatch.setattr(llm, "call_llm", quota_gone)
    result = llm.correct_text(fake_pipeline_result())
    assert result["source"] == "fallback"
    assert len(calls) == 1                # stopped immediately


def test_missing_api_key_is_not_retried(monkeypatch, isolated):
    calls = []

    def no_key(prompt, config):
        calls.append(1)
        raise RuntimeError("GEMINI_API_KEY not found.")

    monkeypatch.setattr(llm, "call_llm", no_key)
    result = llm.correct_text(fake_pipeline_result())
    assert result["source"] == "fallback"
    assert len(calls) == 1


def test_garbage_reply_falls_back(monkeypatch, isolated):
    monkeypatch.setattr(llm, "call_llm", lambda prompt, config: "Sure! Here is your text.")
    result = llm.correct_text(fake_pipeline_result())
    assert result["source"] == "fallback"
    assert isolated == {}