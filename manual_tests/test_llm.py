"""
test_llm.py
Manual test for Phase 10. Run from the PROJECT ROOT with:
    python -m manual_tests.test_llm
Tests 2-4 make real API calls (a few seconds each).
"""

from src.llm import build_prompt, correct_text, fallback_correction
from src.pipeline import run_pipeline


def show(text):
    print("INPUT     :", text)
    report = run_pipeline(text)
    result = correct_text(report)
    print("SOURCE    :", result["source"])
    if result["error"]:
        print("ERROR     :", result["error"])
    print("CORRECTED :", result["corrected_text"])
    for change in result["changes"]:
        print(f"  - {change.get('original')} -> {change.get('corrected')} "
              f"[{change.get('type')}] {change.get('explanation')}")
    print("SUMMARY   :", result["summary"])
    print("=" * 60)


# ---- Test 1: build the prompt WITHOUT calling the API ----
print("=== TEST 1: The exact prompt the LLM receives ===")
report = run_pipeline("I recieved teh letter yesterday. She go to school evry day.")
prompt = build_prompt(report["prompt_fields"])
print(prompt)
print(f"\n(prompt length: {len(prompt)} characters, roughly {len(prompt) // 4} tokens)")
print("Any unfilled placeholders left?",
      any(p in prompt for p in ["{original_text}", "{spelling_errors}", "{pos_tags}",
                                "{ngram_analysis}", "{dependency_analysis}", "{normalized_text}"]))
print("=" * 60)

# ---- Test 2: spelling + grammar ----
print("=== TEST 2: Spelling and grammar errors ===")
show("I recieved teh letter yesterday. She go to school evry day.")

# ---- Test 3: real-word errors (dictionary cannot see these) ----
print("=== TEST 3: Real-word errors ===")
show("I want to by a knew car.")

# ---- Test 4: already-correct text (should change nothing) ----
print("=== TEST 4: Correct text ===")
show("The boy eats red apples.")

# ---- Test 5: the fallback, without any API ----
print("=== TEST 5: Fallback (no API involved) ===")
text, changes = fallback_correction(run_pipeline("I recieved teh letter."))
print("Fallback result:", text)
print("Changes:", changes)
print("=" * 60)

# ---- Try your own ----
print("=== TRY YOUR OWN TEXT ===")
show(input("Enter some text: "))