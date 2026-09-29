# check_setup.py: quick test that all libraries are installed correctly

import nltk
import spacy
import streamlit
import yaml
import dotenv

print("nltk version:", nltk.__version__)
print("spacy version:", spacy.__version__)
print("streamlit version:", streamlit.__version__)

# Test spaCy model
nlp = spacy.load("en_core_web_sm")
doc = nlp("She went to college.")
print("spaCy works. Tokens:", [token.text for token in doc])

# Test NLTK data
from nltk.tokenize import sent_tokenize
print("NLTK works. Sentences:", sent_tokenize("Hello there. How are you?"))

from nltk.corpus import words
print("Word list size:", len(words.words()))

print("\nALL GOOD. Setup is complete.")