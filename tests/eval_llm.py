"""
eval_llm.py
Phase 12: evaluate the LLM correction on data/test_sentences.csv.
Run from the project root:  python -m tests.eval_llm
Uses the API (cached answers are free). Safe to run again: finished rows come
from the cache, and the script stops cleanly when the daily quota is used up.
"""

import csv
import re
import time
from collections import defaultdict

from src.llm import correct_text, is_daily_quota_error, load_llm_config
from src.pipeline import run_pipeline

TEST_FILE = "data/test_sentences.csv"
OUTPUT_FILE = "data/llm_eval_output.csv"
PAUSE_SECONDS = 6   # pause after each real (non-cached) API call
MAX_ROUNDS = 2            # tries per row before giving up on it for now
WAIT_AFTER_FAILURE = 30   # seconds to wait after a failed attempt


def same_text(a, b):
    """Exact match, ignoring extra spaces."""
    return " ".join(a.split()) == " ".join(b.split())


with open(TEST_FILE, "r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

print("Model:", load_llm_config()["model"])
print("Rows :", len(rows))
print()

completed, pending = [], []
stopped_reason = None

for row in rows:
    report = run_pipeline(row["input"])
    result = None

    for round_number in range(1, MAX_ROUNDS + 1):
        result = correct_text(report)
        if result["source"] == "llm":
            break
        error_text = str(result["error"])
        quota = re.search(r"'quotaId': '([^']+)'", error_text)
        print(f"Row {row['id']:>2}: attempt {round_number} failed: {error_text[:28]}"
              + (f"  [{quota.group(1)}]" if quota else ""))
        if is_daily_quota_error(error_text):
            break
        if round_number < MAX_ROUNDS:
            time.sleep(WAIT_AFTER_FAILURE)

    if result["source"] != "llm":
        pending.append(row["id"])
        print(f"Row {row['id']:>2}: PENDING")
        if is_daily_quota_error(str(result["error"])):
            stopped_reason = "daily quota used up"
            break
        time.sleep(PAUSE_SECONDS)   # pause after failures too
        continue

    match = same_text(result["corrected_text"], row["expected"])
    completed.append({**row, "output": result["corrected_text"], "match": match,
                      "changed": not same_text(result["corrected_text"], row["input"])})
    print(f"Row {row['id']:>2}: {'MATCH ' if match else 'DIFFER'} "
          f"({'cache' if result.get('cached') else 'API'})")
    if not result.get("cached"):
        time.sleep(PAUSE_SECONDS)

# Rows never reached because we stopped early are also pending
if stopped_reason:
    done_ids = {r["id"] for r in completed} | set(pending)
    pending += [r["id"] for r in rows if r["id"] not in done_ids]

print()
print("=== RESULTS (LLM answers only) ===")
print(f"Completed : {len(completed)}/{len(rows)}")
print(f"Pending   : {pending or 'none'}" + (f"  (stopped: {stopped_reason})" if stopped_reason else ""))

if completed:
    matches = sum(r["match"] for r in completed)
    print(f"Exact match: {matches}/{len(completed)}")

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
    any_diff = False
    for r in completed:
        if not r["match"]:
            any_diff = True
            print(f"Row {r['id']} [{r['category']}]")
            print(f"   input   : {r['input']}")
            print(f"   expected: {r['expected']}")
            print(f"   got     : {r['output']}")
    if not any_diff:
        print("None")

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "category", "input", "expected", "output", "match"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)
    print(f"\nSaved to {OUTPUT_FILE}")