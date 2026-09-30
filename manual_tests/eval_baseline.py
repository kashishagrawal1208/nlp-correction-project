"""
eval_baseline.py
Baseline experiment: the same model, temperature, rules and output format as the
main system, but WITHOUT the NLP evidence (prompts/baseline_prompt.txt).
Run from the project root:  python -m manual_tests.eval_baseline
Uses the API (25 calls at most; answers are cached, so re-running is free).
"""

import csv
import time
from collections import defaultdict

from src.llm import (build_prompt, cache_key, call_llm, is_daily_quota_error,
                     load_cache, load_llm_config, load_prompt_template,
                     parse_llm_json, save_to_cache)
from src.normalization import normalize_text

TEST_FILE = "data/test_sentences.csv"
PROMPT_FILE = "prompts/baseline_prompt.txt"
PAUSE_SECONDS = 6
MAX_ROUNDS = 2
WAIT_AFTER_FAILURE = 30


def same_text(a, b):
    """Exact match, ignoring extra spaces."""
    return " ".join(a.split()) == " ".join(b.split())


config = load_llm_config()
template = load_prompt_template(PROMPT_FILE)
with open(TEST_FILE, "r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

print("Model :", config["model"])
print("Prompt:", PROMPT_FILE, "(no NLP evidence)")
print("Rows  :", len(rows))
print()

completed, pending = [], []
stopped = False

for row in rows:
    prompt = build_prompt({"normalized_text": normalize_text(row["input"])}, template)
    key = cache_key(prompt, config)
    data, from_cache, error = load_cache().get(key), True, None

    if data is None:
        from_cache = False
        for round_number in range(1, MAX_ROUNDS + 1):
            try:
                data = parse_llm_json(call_llm(prompt, config))
                save_to_cache(key, data)
                break
            except Exception as exc:
                error = str(exc)
                print(f"Row {row['id']:>2}: attempt {round_number} failed: {error[:60]}")
                if is_daily_quota_error(exc):
                    stopped = True
                    break
                if round_number < MAX_ROUNDS:
                    time.sleep(WAIT_AFTER_FAILURE)
        if data is None:
            pending.append(row["id"])
            print(f"Row {row['id']:>2}: PENDING")
            if stopped:
                break
            time.sleep(PAUSE_SECONDS)
            continue

    match = same_text(data["corrected_text"], row["expected"])
    completed.append({**row, "output": data["corrected_text"], "match": match,
                      "changed": not same_text(data["corrected_text"], row["input"])})
    print(f"Row {row['id']:>2}: {'MATCH ' if match else 'DIFFER'} ({'cache' if from_cache else 'API'})")
    if not from_cache:
        time.sleep(PAUSE_SECONDS)

if stopped:
    done = {r["id"] for r in completed} | set(pending)
    pending += [r["id"] for r in rows if r["id"] not in done]

print()
print("=== BASELINE RESULTS (no NLP evidence) ===")
print(f"Completed : {len(completed)}/{len(rows)}")
print(f"Pending   : {pending or 'none'}")

if completed:
    print(f"Exact match: {sum(r['match'] for r in completed)}/{len(completed)}")
    by_category = defaultdict(lambda: [0, 0])
    for r in completed:
        by_category[r["category"]][1] += 1
        by_category[r["category"]][0] += r["match"]
    print("\nBy category (exact match):")
    for category, (good, total) in sorted(by_category.items()):
        print(f"  {category:10s} {good}/{total}")
    correct_rows = [r for r in completed if r["category"] == "correct"]
    if correct_rows:
        changed = sum(r["changed"] for r in correct_rows)
        print(f"\nOver-correction: {changed}/{len(correct_rows)} already-correct sentences were changed")

    print("\n=== DIFFERENCES (review each by hand) ===")
    differences = [r for r in completed if not r["match"]]
    for r in differences:
        print(f"Row {r['id']} [{r['category']}]")
        print(f"   input   : {r['input']}")
        print(f"   expected: {r['expected']}")
        print(f"   got     : {r['output']}")
    if not differences:
        print("None")
        