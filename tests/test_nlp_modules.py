"""
test_nlp_modules.py
Automated tests for the traditional NLP modules. No API calls.
Run from the project root:  python -m pytest -q
"""

from src.cky import parse_sentence
from src.ngrams import bigram_prob_mle, bigram_prob_smoothed, rank_candidates
from src.normalization import normalize_text
from src.parser import analyze_sentence
from src.spelling import edit_distance, get_candidates, is_misspelled
from src.tokenizer import split_sentences, tokenize_text


# ---------- Normalization ----------

def test_normalize_fixes_spacing_and_punctuation():
    messy = "I went   to college yesterday .It was raining!!!"
    assert normalize_text(messy) == "I went to college yesterday. It was raining!"


def test_normalize_keeps_decimal_numbers():
    assert normalize_text("It costs 3.5 rupees,ok") == "It costs 3.5 rupees, ok"


def test_normalize_keeps_letter_case():
    assert normalize_text("Apple and US") == "Apple and US"


# ---------- Tokenization ----------

def test_sentence_split_handles_abbreviation():
    sentences = split_sentences("Dr. Smith arrived. He sat down.")
    assert sentences == ["Dr. Smith arrived.", "He sat down."]


def test_tokenize_separates_words_and_punctuation():
    result = tokenize_text("I went to college yesterday. It was raining.")
    assert len(result) == 2
    assert result[0]["words"] == ["I", "went", "to", "college", "yesterday"]
    assert result[0]["tokens"][-1] == "."


# ---------- Spelling ----------

def test_edit_distance_known_examples():
    assert edit_distance("teh", "the") == 1          # one transposition
    assert edit_distance("kitten", "sitting") == 3   # textbook example


def test_misspelled_word_is_detected():
    assert is_misspelled("recieved") is True


def test_correct_word_is_not_flagged():
    assert is_misspelled("school") is False


def test_top_candidate_for_common_typo():
    assert get_candidates("recieved")[0]["word"] == "received"


# ---------- N-grams ----------

def test_unsmoothed_is_zero_but_smoothed_is_positive():
    assert bigram_prob_mle("the", "zzzz") == 0.0
    assert bigram_prob_smoothed("the", "zzzz") > 0.0


def test_context_prefers_the_over_ten_and_tea():
    ranked = rank_candidates("received", ["the", "ten", "tea"], "letter")
    assert ranked[0][0] == "the"


# ---------- Grammar (spaCy + agreement rule) ----------

def test_agreement_error_is_warned():
    issues = analyze_sentence("She go to school every day.")["possible_issues"]
    assert any("agreement" in issue for issue in issues)


def test_correct_sentence_gets_no_warning():
    assert analyze_sentence("The boy eats red apples.")["possible_issues"] == []


# ---------- CFG / CKY ----------

def test_cky_accepts_grammatical_and_rejects_wrong_order():
    assert parse_sentence("The boy eats apples.")["accepted"] is True
    assert parse_sentence("Boy the eats apples.")["accepted"] is False


def test_cky_counts_two_parses_for_ambiguous_sentence():
    result = parse_sentence("The boy eats apples in the park.")
    assert result["num_parses"] == 2