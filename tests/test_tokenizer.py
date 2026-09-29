"""
test_tokenizer.py
Manual test for Phase 3. Run from the PROJECT ROOT with:
    python -m tests.test_tokenizer
"""

from src.normalization import normalize_text, simple_sentence_split
from src.tokenizer import split_sentences, split_words, tokenize_text

# ---- Test 1: the example from the project brief ----
print("=== TEST 1: Basic tokenization ===")
text = "I went to college yesterday. It was raining."
for number, item in enumerate(tokenize_text(text), start=1):
    print(f"Sentence {number}: {item['sentence']}")
    print(f"  Tokens: {item['tokens']}")
    print(f"  Words : {item['words']}")
print()

# ---- Test 2: contractions and punctuation ----
print("=== TEST 2: Contractions ===")
print(split_words("I don't know, but she can't go."))
print()

# ---- Test 3: rule-based (Phase 2) vs NLTK Punkt ----
print("=== TEST 3: Our rule vs NLTK Punkt ===")
tricky = "Dr. Smith arrived. He sat down."
print("Text          :", tricky)
print("Our rule      :", simple_sentence_split(tricky))
print("NLTK (Punkt)  :", split_sentences(tricky))
print()

# ---- Test 4: normalization THEN tokenization (mini pipeline) ----
print("=== TEST 4: Normalize, then tokenize ===")
messy = "I recieved   teh letter yesterday .She go to school ,every day!!!"
clean = normalize_text(messy)
print("Messy     :", repr(messy))
print("Normalized:", repr(clean))
for number, item in enumerate(tokenize_text(clean), start=1):
    print(f"Sentence {number}: {item['sentence']}")
    print(f"  Words: {item['words']}")
print()

# ---- Try your own ----
print("=== TRY YOUR OWN TEXT ===")
user_text = input("Enter some text: ")
for number, item in enumerate(tokenize_text(normalize_text(user_text)), start=1):
    print(f"Sentence {number}: {item['sentence']}")
    print(f"  Tokens: {item['tokens']}")
    print(f"  Words : {item['words']}")