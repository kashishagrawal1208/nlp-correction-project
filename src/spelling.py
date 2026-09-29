"""
spelling.py
Phase 4: Spelling error detection and candidate correction.

Method:
  1. Lexicon lookup   -> detect words that are not in the dictionary
  2. Edit operations  -> generate candidate words (1 or 2 edits away)
  3. Ranking          -> smaller edit distance first, then higher word frequency
"""

from collections import Counter

import yaml
from nltk.corpus import brown
from nltk.corpus import words as nltk_words

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# These are filled once by load_resources() and reused (loading is slow).
_word_counts = None
_lexicon = None


def load_resources():
    """Build the lexicon and word-frequency table (only the first time)."""
    global _word_counts, _lexicon
    if _lexicon is not None:
        return

    # Frequency of every alphabetic word in the Brown corpus (lowercased)
    _word_counts = Counter(w.lower() for w in brown.words() if w.isalpha())

    # Source 1: Brown words seen at least twice (filters one-off typos)
    lexicon = {w for w, count in _word_counts.items() if count >= 2}
    # Source 2: NLTK's big English word list
    lexicon |= {w.lower() for w in nltk_words.words() if w.isalpha()}

    # The word lists contain junk single letters; only "a" and "i" are real words
    lexicon = {w for w in lexicon if len(w) > 1}
    lexicon |= {"a", "i"}
    _lexicon = lexicon


def get_max_edit_distance():
    """Read max_edit_distance from config/config.yaml (default 2)."""
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file)["nlp"]["max_edit_distance"]
    except (OSError, KeyError, TypeError):
        return 2


def edit_distance(a, b):
    """Minimum number of insertions, deletions, substitutions and
    transpositions needed to turn word a into word b.
    Uses dynamic programming: table[i][j] = distance between the first
    i letters of a and the first j letters of b."""
    rows, cols = len(a) + 1, len(b) + 1
    table = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        table[i][0] = i          # turning a[:i] into "" needs i deletions
    for j in range(cols):
        table[0][j] = j          # turning "" into b[:j] needs j insertions

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,         # deletion
                table[i][j - 1] + 1,         # insertion
                table[i - 1][j - 1] + cost,  # substitution (or match)
            )
            # transposition: swapped neighbouring letters ("teh" -> "the")
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                table[i][j] = min(table[i][j], table[i - 2][j - 2] + 1)

    return table[-1][-1]


def edits_one_away(word):
    """Every string that is exactly ONE edit away from `word`."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [left + right[1:] for left, right in splits if right]
    transposes = [left + right[1] + right[0] + right[2:]
                  for left, right in splits if len(right) > 1]
    replaces = [left + c + right[1:] for left, right in splits if right for c in LETTERS]
    inserts = [left + c + right for left, right in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


def known(candidates):
    """Keep only the candidates that are real words in the lexicon."""
    return {w for w in candidates if w in _lexicon}


def get_candidates(word, max_distance=2, top_n=5):
    """Return up to top_n suggested corrections for a misspelled word."""
    load_resources()
    word = word.lower()

    one_edit = edits_one_away(word)
    pool = known(one_edit)

    if not pool and max_distance >= 2:
        two_edits = {e2 for e1 in one_edit for e2 in edits_one_away(e1)}
        pool = known(two_edits)

    ranked = sorted(
        pool,
        key=lambda w: (edit_distance(word, w), -_word_counts.get(w, 0)),
    )
    return [
        {"word": w, "distance": edit_distance(word, w)}
        for w in ranked[:top_n]
    ]


def is_misspelled(token, is_first_word=False):
    """Decide whether one token looks like a spelling error."""
    load_resources()
    # Skip numbers, contractions ("n't"), hyphenated words, etc.
    if not token.isalpha():
        return False
    # A capitalised word in the middle of a sentence is probably a name
    if token[0].isupper() and not is_first_word:
        return False
    return token.lower() not in _lexicon


def detect_spelling_errors(tokenized):
    """Find spelling errors in the output of tokenizer.tokenize_text().

    Returns a list like:
    [{"word": "recieved", "sentence": 0, "position": 1,
      "candidates": [{"word": "received", "distance": 1}, ...]}]
    """
    max_distance = get_max_edit_distance()
    errors = []
    for sentence_index, item in enumerate(tokenized):
        for position, word in enumerate(item["words"]):
            if is_misspelled(word, is_first_word=(position == 0)):
                errors.append({
                    "word": word,
                    "sentence": sentence_index,
                    "position": position,
                    "candidates": get_candidates(word, max_distance),
                })
    return errors