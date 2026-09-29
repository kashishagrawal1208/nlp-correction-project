"""
ngrams.py
Phase 5: N-gram language model (unigram, bigram, trigram) with add-one smoothing.

The model is trained on the Brown corpus. It answers the question:
"How likely is this word, given the word(s) before it?"
"""

import math
from collections import Counter

from nltk.corpus import brown

START = "<s>"
END = "</s>"

# Filled once by train() and reused.
_unigrams = None
_bigrams = None
_trigrams = None
_vocab_size = 0
_total_words = 0


def train():
    """Count unigrams, bigrams and trigrams in the Brown corpus (first call only)."""
    global _unigrams, _bigrams, _trigrams, _vocab_size, _total_words
    if _unigrams is not None:
        return

    _unigrams, _bigrams, _trigrams = Counter(), Counter(), Counter()

    for sentence in brown.sents():
        # Keep only alphabetic words, lowercased, wrapped in boundary markers
        words = [START, START] + [w.lower() for w in sentence if w.isalpha()] + [END]
        _unigrams.update(words)
        _bigrams.update(zip(words, words[1:]))
        _trigrams.update(zip(words, words[1:], words[2:]))

    _vocab_size = len(_unigrams)
    _total_words = sum(_unigrams.values())


# ---------- Unsmoothed (MLE) probabilities ----------

def unigram_prob(word):
    """P(word) = count(word) / total words. Unsmoothed."""
    train()
    return _unigrams[word] / _total_words


def bigram_prob_mle(prev, word):
    """P(word | prev) = count(prev word) / count(prev). Unsmoothed.
    Returns 0.0 for unseen bigrams: this is the zero-probability problem."""
    train()
    if _unigrams[prev] == 0:
        return 0.0
    return _bigrams[(prev, word)] / _unigrams[prev]


# ---------- Smoothed probabilities ----------

def bigram_prob_smoothed(prev, word):
    """Add-one (Laplace) smoothed P(word | prev).
    (count(prev word) + 1) / (count(prev) + V). Never zero."""
    train()
    return (_bigrams[(prev, word)] + 1) / (_unigrams[prev] + _vocab_size)


def trigram_prob_smoothed(prev2, prev1, word):
    """Add-one smoothed P(word | prev2 prev1)."""
    train()
    return (_trigrams[(prev2, prev1, word)] + 1) / (_bigrams[(prev2, prev1)] + _vocab_size)


# ---------- Sentence-level scoring ----------

def sentence_log_prob(words):
    """Log-probability of a list of words under the smoothed bigram model.
    Closer to 0 = more likely. Different sentences of the SAME length are comparable."""
    train()
    words = [START] + [w.lower() for w in words] + [END]
    total = 0.0
    for prev, word in zip(words, words[1:]):
        total += math.log(bigram_prob_smoothed(prev, word))
    return total


def rank_candidates(left_word, candidates, right_word=None):
    """Rank spelling candidates by how well they fit BETWEEN left_word and right_word.

    score = log P(candidate | left) + log P(right | candidate)
    left_word / right_word may be None (start / end of sentence).
    Returns a list of (candidate, score), best first."""
    train()
    left = left_word.lower() if left_word else START
    right = right_word.lower() if right_word else END

    scored = []
    for candidate in candidates:
        score = math.log(bigram_prob_smoothed(left, candidate))
        score += math.log(bigram_prob_smoothed(candidate, right))
        scored.append((candidate, score))
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


# ---------- Report for the pipeline ----------

def ngram_report(words):
    """Summarise n-gram evidence for one sentence (used later in the LLM prompt).

    Flags bigrams that NEVER appeared in the corpus: these are suspicious
    word pairs, which can reveal real-word errors like 'by a knew car'."""
    train()
    lowered = [w.lower() for w in words]
    padded = [START] + lowered + [END]

    unseen = []
    for prev, word in zip(padded, padded[1:]):
        if _bigrams[(prev, word)] == 0:
            unseen.append(f"{prev} {word}")

    return {
        "sentence_log_prob": round(sentence_log_prob(words), 2),
        "avg_log_prob_per_word": round(sentence_log_prob(words) / (len(words) + 1), 2),
        "unseen_bigrams": unseen,
    }