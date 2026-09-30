"""
test_cache.py
Run from the project root:  python -m manual_tests.test_cache
Costs at most ONE API call. The second run of the same sentence must be cached.
"""

import time

from src.llm import correct_text
from src.pipeline import run_pipeline

report = run_pipeline("She go to school every day.")

for attempt in (1, 2):
    start = time.time()
    result = correct_text(report)
    print(f"Run {attempt}: source={result['source']}, cached={result.get('cached', False)}, "
          f"time={time.time() - start:.1f}s")
    print("   corrected:", result["corrected_text"])
    if result["error"]:
        print("   error:", result["error"])