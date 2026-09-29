"""
cky.py
Phase 8: A small context-free grammar (in Chomsky Normal Form) and a
CKY parser. This is an academic demonstration, separate from the main
pipeline.

CKY = fill a triangular table bottom-up; cell (i, j) holds every category
that can produce the words i..j of the sentence.
"""

from collections import defaultdict

from src.tokenizer import is_word, split_words

# Grammar in Chomsky Normal Form:
#   A -> B C     (two non-terminals, uppercase)
#   A -> word    (one terminal, lowercase)
GRAMMAR_TEXT = """
S   -> NP VP
NP  -> Det N
NP  -> NP PP
VP  -> V NP
VP  -> VP PP
PP  -> P NP

Det -> the | a
N   -> boy | boys | girl | dog | park | book | apples
V   -> eats | sees | likes | reads
P   -> in | near | with
NP  -> she | he | they | apples
VP  -> sleeps | runs
"""


def load_grammar(text):
    """Turn the grammar text into two lookup tables.

    lexical[word]     = set of categories for that word
    binary[(B, C)]    = set of categories A such that A -> B C
    """
    lexical = defaultdict(set)
    binary = defaultdict(set)
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        left, right = line.split("->")
        left = left.strip()
        for option in right.split("|"):
            symbols = option.split()
            if len(symbols) == 2:
                binary[(symbols[0], symbols[1])].add(left)
            elif len(symbols) == 1 and symbols[0][0].islower():
                lexical[symbols[0]].add(left)
            else:
                raise ValueError(f"Rule is not in CNF: {line}")
    return lexical, binary


LEXICAL, BINARY = load_grammar(GRAMMAR_TEXT)


def cky_table(words):
    """Fill the CKY table.

    table[i][j] is a dictionary: category -> list of "back-pointers"
    that record HOW that category was built (so we can rebuild trees).
      word rule   : ("word", the_word)
      binary rule : (k, B, C)  meaning span i..k is B and span k..j is C
    """
    n = len(words)
    table = [[{} for _ in range(n + 1)] for _ in range(n + 1)]

    # Step 1: bottom row. Each single word gets its possible categories.
    for i, word in enumerate(words):
        for category in LEXICAL.get(word.lower(), ()):
            table[i][i + 1].setdefault(category, []).append(("word", word))

    # Step 2: longer and longer spans.
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            for k in range(i + 1, j):                 # every split point
                for left in table[i][k]:
                    for right in table[k][j]:
                        for parent in BINARY.get((left, right), ()):
                            table[i][j].setdefault(parent, []).append((k, left, right))
    return table


def count_trees(table, i, j, symbol):
    """How many different parse trees give `symbol` for span i..j?"""
    total = 0
    for back in table[i][j][symbol]:
        if len(back) == 2:                             # word rule
            total += 1
        else:
            k, left, right = back
            total += count_trees(table, i, k, left) * count_trees(table, k, j, right)
    return total


def build_tree(table, i, j, symbol):
    """Rebuild ONE parse tree as nested tuples:
    (category, word)  or  (category, left_subtree, right_subtree)."""
    back = table[i][j][symbol][0]
    if len(back) == 2:
        return (symbol, back[1])
    k, left, right = back
    return (symbol, build_tree(table, i, k, left), build_tree(table, k, j, right))


def tree_to_brackets(tree):
    """Penn-Treebank-style bracket string, e.g. (S (NP ...) (VP ...))."""
    if len(tree) == 2:
        return f"({tree[0]} {tree[1]})"
    return f"({tree[0]} {tree_to_brackets(tree[1])} {tree_to_brackets(tree[2])})"


def tree_to_lines(tree, depth=0):
    """Indented drawing of the tree, as a list of text lines."""
    indent = "    " * depth
    if len(tree) == 2:
        return [f"{indent}{tree[0]} -> {tree[1]}"]
    return ([indent + tree[0]]
            + tree_to_lines(tree[1], depth + 1)
            + tree_to_lines(tree[2], depth + 1))


def parse_sentence(sentence):
    """Parse one sentence with our small grammar.

    Returns a dictionary with: words, unknown_words, accepted (True/False),
    num_parses, tree (first parse or None) and the raw table."""
    words = [t for t in split_words(sentence) if is_word(t)]
    unknown = [w for w in words if w.lower() not in LEXICAL]
    table = cky_table(words)
    n = len(words)

    accepted = n > 0 and "S" in table[0][n]
    return {
        "words": words,
        "unknown_words": unknown,
        "accepted": accepted,
        "num_parses": count_trees(table, 0, n, "S") if accepted else 0,
        "tree": build_tree(table, 0, n, "S") if accepted else None,
        "table": table,
    }


def show_chart(result):
    """Print every non-empty cell of the CKY table, shortest spans first."""
    words, table = result["words"], result["table"]
    for length in range(1, len(words) + 1):
        for i in range(0, len(words) - length + 1):
            j = i + length
            if table[i][j]:
                text = " ".join(words[i:j])
                print(f"  [{i}:{j}] '{text}' -> {sorted(table[i][j])}")