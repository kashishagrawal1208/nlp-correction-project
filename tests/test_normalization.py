"""
test_normalization.py
Quick manual test for Phase 2. Run from the PROJECT ROOT folder with:
    python -m tests.test_normalization
"""

from src.normalization import normalize_text, simple_sentence_split

examples = [
    "I went   to college yesterday .It was raining!!!",
    "hello  world ,this is   a test",
    "\u201cI don\u2019t know,\u201d she said.",
    "Really??? Yes!!!",
    "It costs 3.5 rupees,ok",
]

print("=== NORMALIZATION TESTS ===")
for text in examples:
    print("Before:", repr(text))
    print("After :", repr(normalize_text(text)))
    print()

print("=== SENTENCE SEGMENTATION TESTS ===")
good = "I went to college yesterday. It was raining. Did you go?"
print("Text     :", good)
print("Sentences:", simple_sentence_split(good))
print()

tricky = "Dr. Smith arrived. He sat down."
print("Text     :", tricky)
print("Sentences:", simple_sentence_split(tricky))
print("(Notice the mistake: 'Dr.' was treated as a sentence end!)")
print()

# Interactive part: type your own text
print("=== TRY YOUR OWN TEXT ===")
user_text = input("Enter some messy text: ")
print("Normalized:", normalize_text(user_text))
print("Sentences :", simple_sentence_split(normalize_text(user_text)))