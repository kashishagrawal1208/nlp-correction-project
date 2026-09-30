"""
inspect_grammar.py
Phase 12: look at how spaCy parsed the two sentences our agreement check missed.
Run from the project root:  python -m manual_tests.inspect_grammar
Makes NO API calls.
"""

from src.pos_tagger import get_spacy_model

nlp = get_spacy_model()

for sentence in ["My brother don't like coffee.", "The childrens is playing in the gardan."]:
    print("SENTENCE:", sentence)
    print(f"  {'WORD':10s} {'TAG':6s} {'RELATION':10s} HEAD")
    for token in nlp(sentence):
        print(f"  {token.text:10s} {token.tag_:6s} {token.dep_:10s} {token.head.text}")
    print()