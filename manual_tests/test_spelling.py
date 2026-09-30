"""
test_spelling.py
Manual test for Phase 4. Run from the PROJECT ROOT with:
    python -m manual_tests.test_spelling
The first run takes a few seconds (building the lexicon).
"""

from src.normalization import normalize_text
from src.tokenizer import tokenize_text
from src.spelling import edit_distance, detect_spelling_errors, get_candidates

# ---- Test 1: edit distance ----
print("=== TEST 1: Edit distance ===")
for a, b in [("teh", "the"), ("recieved", "received"), ("cat", "cart"), ("kitten", "sitting")]:
    print(f"{a} -> {b}: {edit_distance(a, b)}")
print()

# ---- Test 2: candidates for single misspelled words ----
print("=== TEST 2: Candidates ===")
for word in ["recieved", "teh", "definately", "wich", "beutiful"]:
    print(word, "->", [c["word"] for c in get_candidates(word)])
print()

# ---- Test 3: a full sentence ----
print("=== TEST 3: Full sentence ===")
text = "I recieved teh letter yesterday. She go to school evry day."
tokenized = tokenize_text(normalize_text(text))
for error in detect_spelling_errors(tokenized):
    suggestions = [c["word"] for c in error["candidates"]]
    print(f"'{error['word']}' (sentence {error['sentence']}, word {error['position']}) -> {suggestions}")
print()

# ---- Test 4: what a dictionary CANNOT catch ----
print("=== TEST 4: Real-word errors (should find NOTHING) ===")
text = "I want to by a knew car. She go to school."
tokenized = tokenize_text(normalize_text(text))
print("Errors found:", detect_spelling_errors(tokenized))
print("(Every word is a valid English word, so the dictionary is blind here.)")
print()

# ---- Try your own ----
print("=== TRY YOUR OWN TEXT ===")
user_text = input("Enter some text: ")
tokenized = tokenize_text(normalize_text(user_text))
errors = detect_spelling_errors(tokenized)
if not errors:
    print("No spelling errors detected.")
for error in errors:
    print(error["word"], "->", [(c["word"], c["distance"]) for c in error["candidates"]])