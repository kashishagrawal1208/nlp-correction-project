"""
test_ngrams.py
Manual test for Phase 5. Run from the PROJECT ROOT with:
    python -m tests.test_ngrams
Training on Brown takes several seconds the first time.
"""

from src.ngrams import (
    bigram_prob_mle,
    bigram_prob_smoothed,
    ngram_report,
    rank_candidates,
    sentence_log_prob,
    trigram_prob_smoothed,
    unigram_prob,
)

# ---- Test 1: unigram, bigram, trigram ----
print("=== TEST 1: N-grams of 'the cat is' ===")
print("Unigrams:", ["the", "cat", "is"])
print("Bigrams :", ["the cat", "cat is"])
print("Trigrams:", ["the cat is"])
print()

# ---- Test 2: unsmoothed vs smoothed ----
print("=== TEST 2: Unsmoothed vs smoothed ===")
print(f"Unigram P('the')               = {unigram_prob('the'):.5f}")
print(f"MLE     P(of | the)            = {bigram_prob_mle('the', 'of'):.5f}")
print(f"Smooth  P(of | the)            = {bigram_prob_smoothed('the', 'of'):.7f}")
print(f"MLE     P(banana | the)        = {bigram_prob_mle('the', 'banana'):.5f}")
print(f"Smooth  P(banana | the)        = {bigram_prob_smoothed('the', 'banana'):.7f}")
print(f"MLE     P(zzzz | the)          = {bigram_prob_mle('the', 'zzzz'):.5f}   <- zero-probability problem")
print(f"Smooth  P(zzzz | the)          = {bigram_prob_smoothed('the', 'zzzz'):.7f}   <- fixed by smoothing")
print(f"Trigram smooth P(of | in, the) = {trigram_prob_smoothed('in', 'the', 'of'):.7f}")
print()

# ---- Test 3: which sentence is more likely? ----
print("=== TEST 3: Sentence scores (closer to 0 = more likely) ===")
good = ["i", "want", "to", "buy", "a", "new", "car"]
bad = ["i", "want", "to", "by", "a", "knew", "car"]
print("Good:", round(sentence_log_prob(good), 2))
print("Bad :", round(sentence_log_prob(bad), 2))
print()

# ---- Test 4: ranking spelling candidates in context ----
print("=== TEST 4: Ranking candidates in context ===")
print("'i recieved ___ letter'  candidates for teh: the / ten / tea")
for candidate, score in rank_candidates("received", ["the", "ten", "tea"], "letter"):
    print(f"  {candidate:5s} {score:.2f}")
print()

# ---- Test 5: n-gram report ----
print("=== TEST 5: N-gram report ===")
print(ngram_report(["i", "want", "to", "by", "a", "knew", "car"]))
print()

# ---- Try your own ----
print("=== TRY YOUR OWN ===")
user_text = input("Enter a sentence (words only, no punctuation): ")
print(ngram_report(user_text.split()))