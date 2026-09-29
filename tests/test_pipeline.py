"""
test_pipeline.py
Manual test for Phase 9. Run from the PROJECT ROOT with:
    python -m tests.test_pipeline
The first run takes a while (loads Brown corpus and spaCy).
"""

import time

from src.pipeline import run_pipeline


def show(text):
    start = time.time()
    result = run_pipeline(text)
    print("ORIGINAL   :", result["original_text"])
    print("NORMALIZED :", result["normalized_text"])
    print("SENTENCES  :", [t["sentence"] for t in result["tokenized"]])

    print("\nSPELLING ERRORS (best in context first):")
    if not result["spelling_errors"]:
        print("  none")
    for e in result["spelling_errors"]:
        print(f"  {e['word']} -> best in context: {e['best_in_context']}")
        for c in e["candidates"][:3]:
            print(f"      {c['word']:12s} distance={c['distance']}  context_score={c['context_score']}")

    print("\n--- PROMPT FIELDS (what the LLM will receive) ---")
    for name, value in result["prompt_fields"].items():
        print(f"[{name}]")
        print(value)
        print()
    print(f"(pipeline took {time.time() - start:.1f} seconds)")
    print("=" * 60)


print("=== TEST 1: Spelling + grammar errors ===")
show("I recieved teh letter yesterday .She go to school evry day!!!")

print("=== TEST 2: Real-word errors (no spelling errors expected) ===")
show("I want to by a knew car.")

print("=== TEST 3: Correct text ===")
show("The boy eats red apples.")

print("=== TRY YOUR OWN TEXT ===")
show(input("Enter some text: "))