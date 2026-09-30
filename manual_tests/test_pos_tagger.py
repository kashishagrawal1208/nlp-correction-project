"""
test_pos_tagger.py
Manual test for Phase 6. Run from the PROJECT ROOT with:
    python -m manual_tests.test_pos_tagger
"""

from src.normalization import normalize_text
from src.tokenizer import tokenize_text
from src.pos_tagger import compare_taggers, tag_text


def show(text):
    tokenized = tokenize_text(normalize_text(text))
    for tagged in tag_text(tokenized):
        print("Sentence:", tagged["sentence"])
        print(f"  {'WORD':12s} {'UPOS':7s} {'PTB(spaCy)':11s}")
        for item in tagged["spacy"]:
            print(f"  {item['word']:12s} {item['upos']:7s} {item['ptb']:11s}")
        print("  NLTK tags  :", tagged["nltk"])
        print("  Disagreements:", compare_taggers(tagged))
        print()


# ---- Test 1: a correct sentence ----
print("=== TEST 1: Correct sentence ===")
show("She went to college yesterday.")

# ---- Test 2: grammar errors ----
print("=== TEST 2: Subject-verb agreement error ===")
show("She go to school every day.")

print("=== TEST 3: Correct version, for comparison ===")
show("She goes to school every day.")

# ---- Test 4: ambiguity ----
print("=== TEST 4: Ambiguity (same word, different tag) ===")
show("I read books. I read it yesterday.")

# ---- Try your own ----
print("=== TRY YOUR OWN TEXT ===")
show(input("Enter some text: "))
