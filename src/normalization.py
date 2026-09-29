"""
normalization.py
Phase 2: Text normalization and simple sentence segmentation.

Normalization = cleaning messy text into a consistent form.
We keep the original letter case on purpose (see lowercase_for_lookup below).
"""

import re


def normalize_text(text):
    """Clean spacing and punctuation. Does NOT change letter case or words."""

    # 1. Convert curly quotes/apostrophes to plain ones (’ -> ')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # 2. Collapse any run of whitespace (spaces, tabs, newlines) into one space
    text = re.sub(r"\s+", " ", text)

    # 3. Remove the space BEFORE punctuation: "hello ," -> "hello,"
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # 4. Collapse repeated punctuation: "!!!" -> "!", "?????" -> "?"
    text = re.sub(r"([!?])\1+", r"\1", text)
    #    Very long dots "......" -> "..." (a normal ellipsis)
    text = re.sub(r"\.{4,}", "...", text)

    # 5. Add a missing space AFTER punctuation: "hello,world" -> "hello, world"
    #    (only before a letter, so numbers like 1,000 are left alone)
    text = re.sub(r"([,;:!?])(?=[A-Za-z])", r"\1 ", text)
    #    For periods we are more careful: only "lowercase letter + . + Capital"
    #    so "3.5" and "e.g." are not damaged. "yesterday.It" -> "yesterday. It"
    text = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", text)

    # 6. Remove spaces at the very start/end
    return text.strip()


def lowercase_for_lookup(word):
    """Lowercase a word ONLY for temporary comparison (e.g. dictionary lookup).
    The original text is never changed by this function."""
    return word.lower()


def simple_sentence_split(text):
    """A simple RULE-BASED sentence segmenter.
    Rule: split after . ! or ? when followed by a space and a capital letter.
    Limitation: it wrongly splits after abbreviations like 'Dr.' (we'll compare
    this with NLTK's trained segmenter in Phase 3)."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    return [p for p in parts if p]