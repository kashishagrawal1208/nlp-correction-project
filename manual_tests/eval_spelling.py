"""
eval_spelling.py
Phase 12: evaluate spelling detection and candidate ranking on the test set.
Run from the project root:  python -m manual_tests.eval_spelling
Makes NO API calls.
"""

import csv

from src.normalization import normalize_text
from src.pipeline import rank_errors_in_context
from src.spelling import detect_spelling_errors
from src.tokenizer import tokenize_text

TEST_FILE = "data/test_sentences.csv"


def load_rows():
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_gold(cell):
    """'recieved>received;teh>the' -> {'recieved': 'received', 'teh': 'the'}"""
    gold = {}
    for pair in cell.split(";"):
        if ">" in pair:
            wrong, right = pair.split(">")
            gold[wrong.strip().lower()] = right.strip().lower()
    return gold


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


rows = load_rows()
tp = fp = fn = 0
top1_edit = top1_context = top3_edit = 0
found = 0
problems = []

for row in rows:
    tokenized = tokenize_text(normalize_text(row["input"]))
    errors = detect_spelling_errors(tokenized)
    ranked = rank_errors_in_context(tokenized, errors)
    gold = parse_gold(row["misspelled"])

    flagged = {e["word"].lower() for e in errors}

    for word in flagged - set(gold):
        fp += 1
        problems.append(f"Row {row['id']}: FALSE POSITIVE, flagged '{word}' but it is not a gold error")
    for word in set(gold) - flagged:
        fn += 1
        problems.append(f"Row {row['id']}: MISSED, '{word}' was not flagged (not in gold detections)")

    for error, ctx in zip(errors, ranked):
        word = error["word"].lower()
        if word not in gold:
            continue
        tp += 1
        found += 1
        right = gold[word]
        edit_list = [c["word"] for c in error["candidates"]]
        if edit_list[:1] == [right]:
            top1_edit += 1
        if right in edit_list[:3]:
            top3_edit += 1
        if ctx["best_in_context"] == right:
            top1_context += 1
        else:
            problems.append(
                f"Row {row['id']}: '{word}' expected '{right}', edit-distance top = "
                f"{edit_list[:1]}, context top = {ctx['best_in_context']}")

precision = ratio(tp, tp + fp)
recall = ratio(tp, tp + fn)
f1 = ratio(2 * precision * recall, precision + recall)

print("=== SPELLING DETECTION (non-word errors) ===")
print(f"Sentences tested : {len(rows)}")
print(f"True positives   : {tp}")
print(f"False positives  : {fp}")
print(f"False negatives  : {fn}")
print(f"Precision        : {precision:.2f}")
print(f"Recall           : {recall:.2f}")
print(f"F1               : {f1:.2f}")
print()
print("=== CANDIDATE QUALITY (for the errors that were found) ===")
print(f"Errors found and scored           : {found}")
print(f"Top-1 correct, edit distance only : {top1_edit}/{found}")
print(f"Top-3 contains correct word       : {top3_edit}/{found}")
print(f"Top-1 correct, N-gram context     : {top1_context}/{found}")
print()
print("=== DETAILS: problems to look at ===")
print("\n".join(problems) if problems else "None")