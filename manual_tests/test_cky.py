"""
test_cky.py
Manual test for Phase 8. Run from the PROJECT ROOT with:
    python -m manual_tests.test_cky
"""

from src.cky import parse_sentence, show_chart, tree_to_brackets, tree_to_lines


def show(sentence, chart=False):
    result = parse_sentence(sentence)
    print("Sentence      :", sentence)
    print("Accepted      :", result["accepted"])
    if result["unknown_words"]:
        print("Unknown words :", result["unknown_words"])
    if chart:
        print("CKY chart (span -> possible categories):")
        show_chart(result)
    if result["accepted"]:
        print("Number of parses:", result["num_parses"])
        print("Brackets      :", tree_to_brackets(result["tree"]))
        print("Tree:")
        for line in tree_to_lines(result["tree"]):
            print("   ", line)
    print()


print("=== TEST 1: The example from the project brief (with chart) ===")
show("The boy eats apples.", chart=True)

print("=== TEST 2: Ambiguity (prepositional phrase attachment) ===")
show("The boy eats apples in the park.")

print("=== TEST 3: Word order is wrong, so the grammar REJECTS it ===")
show("Boy the eats apples.")

print("=== TEST 4: Word not in our small grammar ===")
show("She goes to college.")

print("=== TEST 5: Limitation, a plain CFG ignores agreement ===")
show("The boys eats apples.")

print("=== TRY YOUR OWN SENTENCE ===")
print("Words available: the a | boy boys girl dog park book apples |")
print("                 eats sees likes reads | in near with | she he they | sleeps runs")
show(input("Enter a sentence: "))