"""
test_parser.py
Manual test for Phase 7. Run from the PROJECT ROOT with:
    python -m manual_tests.test_parser
"""

from src.normalization import normalize_text
from src.tokenizer import tokenize_text
from src.parser import analyze_text


def show(text):
    print("TEXT:", text)
    for report in analyze_text(tokenize_text(normalize_text(text))):
        print("Sentence:", report["sentence"])
        print(f"  {'WORD':10s} {'RELATION':10s} {'HEAD':10s}")
        for row in report["dependencies"]:
            print(f"  {row['word']:10s} {row['relation']:10s} {row['head']:10s}")
        print("  Tree:")
        for line in report["tree"].split("\n"):
            print("    " + line)
        print("  Noun phrases  :", report["noun_phrases"])
        print("  Clauses       :", report["clauses"])
        print("  Possible issues:", report["possible_issues"])
        print()


print("=== TEST 1: Simple correct sentence ===")
show("The boy eats red apples.")

print("=== TEST 2: Agreement error ===")
show("She go to school every day.")

print("=== TEST 3: Correct version ===")
show("She goes to school every day.")

print("=== TEST 4: Plural subject, singular verb ===")
show("The boys goes to the park.")

print("=== TEST 5: Words between subject and verb ===")
show("The boys in the red car goes home.")

print("=== TRY YOUR OWN TEXT ===")
show(input("Enter some text: "))