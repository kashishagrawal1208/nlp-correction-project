"""
pos_tagger.py
Phase 6: Part-of-speech tagging with two taggers.

  - NLTK  : averaged perceptron tagger (statistical), Penn Treebank tags
  - spaCy : neural pipeline, gives both Universal (UPOS) and Penn Treebank tags

POS tagging = labelling each word with its grammatical category.
"""

import nltk
import spacy
from src.config import load_config

_nlp = None  # the spaCy model, loaded once and reused


def get_spacy_model():
       """Load the spaCy English model the first time it is needed."""
       global _nlp
       if _nlp is None:
           model_name = load_config().get("nlp", {}).get("spacy_model", "en_core_web_sm")
           _nlp = spacy.load(model_name)
       return _nlp


def tag_with_nltk(tokens):
    """Tag a list of tokens with NLTK. Returns [(word, PTB tag), ...]."""
    return nltk.pos_tag(tokens)


def tag_with_spacy(sentence):
    """Tag one sentence with spaCy.
    Returns a list of dictionaries with the word, UPOS tag and PTB tag."""
    doc = get_spacy_model()(sentence)
    return [
        {"word": token.text, "upos": token.pos_, "ptb": token.tag_}
        for token in doc
    ]


def tag_text(tokenized):
    """Tag every sentence from tokenizer.tokenize_text().

    Returns a list with one entry per sentence:
    [{"sentence": "She go home.",
      "spacy": [{"word": "She", "upos": "PRON", "ptb": "PRP"}, ...],
      "nltk":  [("She", "PRP"), ...]}]
    """
    results = []
    for item in tokenized:
        results.append({
            "sentence": item["sentence"],
            "spacy": tag_with_spacy(item["sentence"]),
            "nltk": tag_with_nltk(item["tokens"]),
        })
    return results


def compare_taggers(tagged_sentence):
    """Find words where NLTK and spaCy disagree on the Penn Treebank tag.
    Disagreement often points to ambiguous or error-containing words."""
    disagreements = []
    for spacy_item, (nltk_word, nltk_tag) in zip(tagged_sentence["spacy"], tagged_sentence["nltk"]):
        if spacy_item["word"] == nltk_word and spacy_item["ptb"] != nltk_tag:
            disagreements.append({
                "word": nltk_word,
                "spacy": spacy_item["ptb"],
                "nltk": nltk_tag,
            })
    return disagreements