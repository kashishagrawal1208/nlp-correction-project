"""
eval_grammar.py
Phase 12: evaluate the rule-based agreement check (parser.py).
Run from the project root:  python -m tests.eval_grammar
Makes NO API calls.
"""

import csv

from src.normalization import normalize_text
from src.parser import analyze_text
from src.tokenizer import tokenize_text

# Decided BEFORE running: rows that contain a subject-verb agreement error.
GOLD_AGREEMENT_ROWS = {"10", "11", "12", "13", "23", "24", "25"}

with open("data/test_sentences.csv", "r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

tp = fp = fn = tn = 0
details = []

for row in rows:
    tokenized = tokenize_text(normalize_text(row["input"]))
    issues = [issue for report in analyze_text(tokenized) for issue in report["possible_issues"]]
    warned = any("agreement" in issue for issue in issues)
    has_error = row["id"] in GOLD_AGREEMENT_ROWS

    if warned and has_error:
        tp += 1
    elif warned and not has_error:
        fp += 1
        details.append(f"Row {row['id']} ({row['category']}): FALSE ALARM -> {issues}")
    elif not warned and has_error:
        fn += 1
        details.append(f"Row {row['id']} ({row['category']}): MISSED -> {row['input']}")
    else:
        tn += 1

    other = [i for i in issues if "agreement" not in i]
    if other:
        details.append(f"Row {row['id']}: other warning -> {other}")


def ratio(a, b):
    return a / b if b else 0.0


precision = ratio(tp, tp + fp)
recall = ratio(tp, tp + fn)

print("=== AGREEMENT CHECK (rule-based) ===")
print(f"Sentences with an agreement error : {len(GOLD_AGREEMENT_ROWS)}")
print(f"Warned correctly (TP)             : {tp}")
print(f"Missed (FN)                       : {fn}")
print(f"False alarms (FP)                 : {fp}")
print(f"Correctly quiet (TN)              : {tn}")
print(f"Precision                         : {precision:.2f}")
print(f"Recall                            : {recall:.2f}")
print()
print("=== DETAILS ===")
print("\n".join(details) if details else "None")