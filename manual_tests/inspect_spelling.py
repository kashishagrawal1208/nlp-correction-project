"""
inspect_spelling.py
Phase 12: look closely at spelling results. Run from the project root:
    python -m manual_tests.inspect_spelling
Makes NO API calls.
"""

import csv

from src.normalization import normalize_text
from src.pipeline import rank_errors_in_context
from src.spelling import detect_spelling_errors
from src.tokenizer import tokenize_text


def show(text):
    tokenized = tokenize_text(normalize_text(text))
    errors = detect_spelling_errors(tokenized)
    ranked = rank_errors_in_context(tokenized, errors)
    print("TEXT:", text)
    if not errors:
        print("  no errors flagged")
    for error, ctx in zip(errors, ranked):
        edit_top3 = [c["word"] for c in error["candidates"][:3]]
        print(f"  flagged '{error['word']}'  edit-distance top 3: {edit_top3}"
              f"  |  context best: {ctx['best_in_context']}")


print("=== PART 1: gold errors where the two rankings DISAGREE ===")
with open("data/test_sentences.csv", "r", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        if not row["misspelled"]:
            continue
        tokenized = tokenize_text(normalize_text(row["input"]))
        errors = detect_spelling_errors(tokenized)
        ranked = rank_errors_in_context(tokenized, errors)
        for error, ctx in zip(errors, ranked):
            edit_top1 = error["candidates"][0]["word"] if error["candidates"] else None
            if edit_top1 != ctx["best_in_context"]:
                print(f"Row {row['id']}: '{error['word']}'  edit-distance top1 = {edit_top1}"
                      f"  context best = {ctx['best_in_context']}")

print()
print("=== PART 2: harder text (not in the test set) ===")
for text in [
    "Priya visited Jaipur yesterday.",
    "I love this app on my iPhone.",
    "Thier friend recieved a pakage.",
    "She is very hapy and exicted.",
    "That was totaly awsome lol.",
    "Teh cat sat on the mat.",
    "He wnats a cofee.",
    "The meeting is at 3pm on Monday.",
]:
    show(text)
    print()