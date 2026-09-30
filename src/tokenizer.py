"""
tokenizer.py
Phase 3: Sentence and word tokenization using NLTK.

Tokenization = breaking text into smaller units (sentences, then words).
"""



from nltk.tokenize import sent_tokenize, word_tokenize


def split_sentences(text):
    """Split text into sentences using NLTK's Punkt model (a trained segmenter)."""
    return sent_tokenize(text)


def split_words(sentence):
    """Split ONE sentence into word and punctuation tokens."""
    return word_tokenize(sentence)


def is_word(token):
    """Return True if the token is a real word (not just punctuation).
    Example: 'college' -> True, '.' -> False, "n't" -> True"""
    return any(character.isalpha() for character in token)


def tokenize_text(text):
    """Full tokenization of a text.

    Returns a list with one dictionary per sentence:
    [
      {"sentence": "I went home.",
       "tokens": ["I", "went", "home", "."],
       "words":  ["I", "went", "home"]},
      ...
    ]
    'tokens' keeps punctuation; 'words' has punctuation removed.
    """
    results = []
    for sentence in split_sentences(text):
        tokens = split_words(sentence)
        words = [token for token in tokens if is_word(token)]
        results.append({
            "sentence": sentence,
            "tokens": tokens,
            "words": words,
        })
    return results