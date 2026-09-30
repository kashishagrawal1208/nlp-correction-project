"""
pipeline.py
Phase 9: Connects all NLP modules into one pipeline.

run_pipeline(text) returns ONE dictionary containing the results of every
stage, plus "prompt_fields": compact text versions used to fill the LLM prompt.
"""

from src.normalization import normalize_text
from src.tokenizer import tokenize_text
from src.spelling import detect_spelling_errors
from src.ngrams import ngram_report, rank_candidates
from src.pos_tagger import compare_taggers, tag_text
from src.parser import analyze_text


# ---------- Stage: rank spelling candidates using N-gram context ----------

def rank_errors_in_context(tokenized, errors):
    """Re-rank each error's candidates by how well they fit between the
    neighbouring words (uses the smoothed bigram model from ngrams.py).

    Adds "left_word", "right_word", "best_in_context", and a "context_score"
    to every candidate (closer to 0 = fits better)."""
    ranked_errors = []
    for error in errors:
        words = tokenized[error["sentence"]]["words"]
        position = error["position"]
        left = words[position - 1] if position > 0 else None
        right = words[position + 1] if position + 1 < len(words) else None

        distances = {c["word"]: c["distance"] for c in error["candidates"]}
        scored = rank_candidates(left, list(distances), right) if distances else []
        candidates = [
            {"word": word, "distance": distances[word], "context_score": round(score, 2)}
            for word, score in scored
        ]
        ranked_errors.append({
            "word": error["word"],
            "sentence": error["sentence"],
            "position": position,
            "left_word": left,
            "right_word": right,
            "candidates": candidates,
            "best_in_context": candidates[0]["word"] if candidates else None,
        })
    return ranked_errors


# ---------- Formatting: turn results into compact text for the LLM ----------

def format_spelling(errors):
    """Turn spelling errors and their context-ranked candidates into prompt text."""
    if not errors:
        return "No non-word spelling errors detected."
    lines = []
    for e in errors:
        options = ", ".join(f"{c['word']} (edit distance {c['distance']})" for c in e["candidates"][:3])
        lines.append(
            f'- "{e["word"]}" in sentence {e["sentence"] + 1}, between '
            f'"{e["left_word"]}" and "{e["right_word"]}". '
            f'Candidates, best fit in context first: {options or "none found"}'
        )
    return "\n".join(lines)


def format_pos(tagged_sentences):
    """Turn POS tags into compact 'word/TAG' text for the prompt."""
    lines = []
    for number, tagged in enumerate(tagged_sentences, start=1):
        tags = " ".join(f"{t['word']}/{t['ptb']}" for t in tagged["spacy"])
        lines.append(f"Sentence {number}: {tags}")
    return "\n".join(lines) or "No tokens."


def format_ngrams(ngram_reports):
    """Turn per-sentence n-gram scores and unseen word pairs into prompt text."""
    lines = []
    for number, report in enumerate(ngram_reports, start=1):
        unseen = ", ".join(report["unseen_bigrams"]) or "none"
        lines.append(
            f"Sentence {number}: average log-probability per word "
            f"{report['avg_log_prob_per_word']} (closer to 0 = more typical); "
            f"word pairs never seen in the reference corpus: {unseen}"
        )
    return "\n".join(lines) or "No sentences."


def format_grammar(grammar_reports):
    """Turn dependency links, verb structure and rule warnings into prompt text."""
    lines = []
    for number, report in enumerate(grammar_reports, start=1):
        deps = ", ".join(
            f"{d['word']}({d['relation']}->{d['head']})"
            for d in report["dependencies"] if d["relation"] != "punct"
        )
        clauses = "; ".join(
            f"verb '{c['verb']}' subjects={c['subjects']} objects={c['objects']}"
            for c in report["clauses"]
        ) or "none"
        issues = "; ".join(report["possible_issues"]) or "none"
        lines.append(
            f"Sentence {number}:\n  dependencies: {deps}\n"
            f"  clauses: {clauses}\n  rule-based warnings: {issues}"
        )
    return "\n".join(lines) or "No sentences."


# ---------- The pipeline ----------

def run_pipeline(text):
    """Run every NLP stage on the input text and return one report dictionary."""
    normalized = normalize_text(text)
    tokenized = tokenize_text(normalized)

    spelling = rank_errors_in_context(tokenized, detect_spelling_errors(tokenized))
    ngrams = [ngram_report(item["words"]) for item in tokenized]
    pos = tag_text(tokenized)
    for tagged in pos:
        tagged["disagreements"] = compare_taggers(tagged)
    grammar = analyze_text(tokenized)

    return {
        "original_text": text,
        "normalized_text": normalized,
        "tokenized": tokenized,
        "spelling_errors": spelling,
        "ngram_analysis": ngrams,
        "pos_tags": pos,
        "grammar_analysis": grammar,
        "prompt_fields": {
            "original_text": text,
            "normalized_text": normalized,
            "spelling_errors": format_spelling(spelling),
            "pos_tags": format_pos(pos),
            "ngram_analysis": format_ngrams(ngrams),
            "dependency_analysis": format_grammar(grammar),
        },
    }