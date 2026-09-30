# A Hybrid NLP-LLM Framework for Context-Aware Spelling and Grammar Correction

Author: Kashish Agarwal

A Python application that analyses English text with **traditional NLP techniques** (normalization, tokenization, spelling detection, n-grams, POS tagging, dependency parsing) and then sends that analysis to an **LLM through an API** for context-aware correction and explanation.

## Problem statement

Ordinary spell checkers work one word at a time. They miss **real-word errors** (`by` instead of `buy`) and **grammar errors** (`She go to school`), because every word is a valid dictionary word. Pure LLM correctors fix these, but they are black boxes: they give no linguistic evidence for their decisions.

## Objective

Build a hybrid pipeline in which classical NLP produces interpretable evidence (candidate corrections, word-sequence statistics, grammatical structure) and an LLM acts as the final decision layer that corrects the text and explains each change.

## Features

- Text normalization (spacing and punctuation cleanup; letter case is preserved on purpose)
- Sentence and word tokenization
- Spelling error detection with a lexicon, edit distance and ranked candidates
- N-gram language model (unigram, bigram, trigram) with add-one smoothing; candidates are re-ranked using the words on both sides
- POS tagging with two taggers (NLTK and spaCy) using the Penn Treebank tagset
- Dependency parsing, verb-argument extraction and a rule-based subject-verb agreement check
- A small CFG with a CKY parser, as a standalone academic demonstration
- LLM correction through the Gemini API, with a prompt built from the NLP report, JSON output that is validated, retries, a fallback when the API fails, and a cache of answers
- Streamlit interface showing the output of every stage
- 34 automated tests (no API calls) and evaluation scripts

## NLP concepts used

| Syllabus topic | Where it appears |
|---|---|
| Text normalization (case study) | `src/normalization.py` |
| Word and sentence tokenization, segmentation | `src/tokenizer.py` (NLTK Punkt); a rule-based splitter is compared in `manual_tests/test_tokenizer.py` |
| Detecting and correcting spelling errors | `src/spelling.py` (lexicon lookup, minimum edit distance by dynamic programming, candidate generation) |
| N-grams, unsmoothed N-grams, smoothing | `src/ngrams.py` (MLE versus add-one smoothing, log probabilities) |
| English word classes, tagsets, POS tagging | `src/pos_tagger.py` (NLTK averaged perceptron, spaCy) |
| Constituency, grammar rules, CFG parsing, CKY | `src/cky.py` (CNF grammar, chart parsing, counting ambiguous parses) |
| Dependency grammar | `src/parser.py` (spaCy dependency parse) |
| Ambiguity | Punctuation in sentence splitting, real-word errors, POS ambiguity, PP-attachment in CKY |
| Finite-state automata, morphology, FSTs | **Conceptual links only.** Regular expressions are used for normalization, and the lexicon is stored in a Python set. **No FSA or FST is implemented.** |

## System architecture

```
User input
   |
   v
Normalization --> Tokenization --> Spelling detection + candidates
                                        |
                                        v
                       N-gram analysis (re-ranks candidates by context)
                                        |
                                        v
                        POS tagging --> Dependency / grammar analysis
                                        |
                                        v
                              NLP analysis report
                                        |
                                        v
              Prompt (prompts/correction_prompt.txt) --> LLM API (Gemini)
                                        |
                                        v
                     JSON validation, retries, fallback, cache
                                        |
                                        v
                  Corrected text + explanation (Streamlit UI)
```

The traditional NLP stages produce **evidence**. The LLM makes the **final judgement**. The CKY module is a separate demonstration and does not feed the pipeline.

## Technologies

Python 3.12 (3.9 to 3.12 should work), NLTK, spaCy (`en_core_web_sm`), Streamlit, PyYAML, python-dotenv, and the `google-genai` SDK for the Gemini API.

## Installation

```bash
git clone https://github.com/kashishagrawal1208/nlp-correction-project.git
cd nlp-correction-project

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng words brown
```

## Configuration

**1. API key.** Create a Gemini API key in Google AI Studio, then:

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

Open `.env` and set `GEMINI_API_KEY=your_real_key`. The `.env` file is listed in `.gitignore` and is never uploaded. The code reads the key from an environment variable and never stores it in source files.

**2. Settings** are in `config/config.yaml`:

| Setting | Meaning |
|---|---|
| `llm.model` | Gemini model name. Google retires models over time; run `python scripts/check_llm_connection.py` to list the models your key can use |
| `llm.temperature` | 0.2, for consistent, repeatable corrections |
| `llm.max_retries` | Extra attempts after a temporary API failure |
| `llm.prompt_file` | Path to the prompt file |
| `nlp.spacy_model` | spaCy model name |
| `nlp.max_edit_distance` | Largest edit distance used for spelling candidates |

## How to run

```bash
python -m streamlit run app.py     # the web interface (opens at http://localhost:8501)
python -m pytest -q                # 34 automated tests, no API calls
python -m manual_tests.test_cky    # CFG + CKY demonstration
```

Use `python -m ...` so that the virtual environment's Python is used.

## LLM integration

The LLM is **not** asked to "correct this sentence". `src/pipeline.py` first produces an NLP report, and `src/llm.py` fills the prompt template with:

- the original and normalized text
- detected spelling errors with candidates ranked by n-gram context
- POS tags (Penn Treebank)
- n-gram evidence (average log-probability, word pairs never seen in the reference corpus)
- dependency links, verb structure and rule-based agreement warnings

The model returns JSON with `corrected_text`, a list of `changes` (original, corrected, type, one-sentence explanation) and a `summary`.

Reliability measures in `src/llm.py`:

- The reply is parsed and validated (required keys and types), and markdown fences are stripped.
- Temporary failures (for example HTTP 503) are retried with exponential backoff.
- If the **daily** free quota is exhausted, the code stops immediately instead of retrying.
- If the API cannot be used, a **rule-based fallback** applies the best in-context spelling candidate. The interface labels this clearly as a fallback.
- Successful answers are cached by a hash of the model, temperature and prompt, so repeated inputs cost no API calls and evaluations are reproducible. Fallback answers are never cached.

## Prompt design

The prompt is in `prompts/correction_prompt.txt`, with placeholders such as `{original_text}`, `{spelling_errors}`, `{pos_tags}`, `{ngram_analysis}` and `{dependency_analysis}`.

| Design choice | Reason |
|---|---|
| Role statement: "final correction stage of a hybrid NLP system" | Defines the LLM as one component of the pipeline |
| Instructions first, input data last | The model reads the goal before the data |
| "Smallest edits", "do not rewrite for style" | Reduces over-correction |
| Each evidence type is explained together with its weakness | The taggers, the parser and the n-gram corpus can be wrong on erroneous text, so evidence is advisory |
| Every difference must be listed in `changes` | Keeps explanations consistent with the output |
| A closed list of `type` values | Consistent, countable output |
| "Output ONLY one JSON object" plus one worked example | Reliable structure at low token cost |

We have **not** measured how much each part of the prompt contributes (no ablation study).

## Sample input and output

Real outputs from the application:

| Input | Corrected output |
|---|---|
| `I recieved teh letter yesterday. She go to school evry day.` | `I received the letter yesterday. She goes to school every day.` |
| `I want to by a knew car.` | `I want to buy a new car.` |

For the second sentence the spell checker finds nothing (both words are in the dictionary). The n-gram model flagged `a knew` and `knew car` as unseen word pairs, but did not flag `to by`.

## Testing

**Automated tests (no API calls):** `python -m pytest -q` runs 34 tests covering normalization, tokenization, spelling, n-grams, agreement checking, CKY, JSON validation, prompt building, the fallback, retries and quota handling.

**Evaluation on our own test set** (`data/test_sentences.csv`, 25 sentences written by the author, with expected answers fixed before running). Full notes are in `docs/evaluation_notes.md`. These numbers describe this small test set only.

| Component | Result |
|---|---|
| Spelling detection (12 misspellings) | 12 found, 0 false alarms |
| Correct word in the top 3 candidates | 12/12 |
| Correct word ranked first: edit distance only versus with n-gram context | 11/12 versus 12/12 (a difference of a single word) |
| Agreement rule, first version | 5 of 7 errors found |
| Agreement rule after checking auxiliary verbs | 7 of 7 found, 0 false alarms. **The rule was changed after seeing these failures on the same test set, so this figure is optimistic** |

**LLM correction** (`gemini-3.1-flash-lite`, temperature 0.2, single run): the corrected text matched the expected sentence exactly for **25 of 25** test sentences, and **0 of 4** already-correct sentences were changed. **Baseline:** the same model with a prompt that has **no NLP evidence** (`prompts/baseline_prompt.txt`) also scored **25 of 25**, so this test set cannot show whether the NLP evidence improves the LLM's corrections. It is small, easy, written by the author, and each condition was run once. We therefore make no claim of an accuracy gain; the value of the hybrid design is the visible intermediate analysis, the fallback when the API fails, and the separately evaluated NLP components. Details are in `docs/evaluation_notes.md`.

What cannot be measured objectively: POS tag accuracy on erroneous text (there are no gold tags), n-gram scores (they are only evidence), and the quality of the explanations.

## Limitations

- The test set is small (25 sentences) and written by the author; the results do not estimate performance on real-world text.
- The lexicon comes from the 1961 Brown corpus and NLTK's word list. Modern words (`app`, `iPhone`, `lol`) and names at the start of a sentence are flagged as errors.
- The n-gram model is trained on a small, old corpus, with crude add-one smoothing. Unseen word pairs are hints, not proof.
- The taggers and the parser were trained on correct English, so they can be misled by misspellings (for example, `evry` changed the parse of its sentence).
- The agreement rule covers present-tense subject-verb agreement only.
- The CFG covers about 30 words and is only a demonstration.
- No FSA or FST is implemented.
- LLM output can vary between runs, free API tiers have daily request limits, and Google retires model names over time.

## Future improvements

Interpolated or Kneser-Ney smoothing, a larger and more modern training corpus, a lexicon that includes modern words, an ablation study of the prompt, a larger and independently written test set, and tense and article checks based on the parse.
a larger, harder and independently written test set, so that the hybrid system can be compared against the baseline.

## Project structure

```
app.py                       Streamlit interface
config/config.yaml           settings
prompts/correction_prompt.txt  the LLM prompt
prompts/baseline_prompt.txt  the same prompt without NLP evidence (baseline experiment)
src/
  config.py                  loads config.yaml once
  normalization.py           text cleaning
  tokenizer.py               sentences and words
  spelling.py                lexicon, edit distance, candidates
  ngrams.py                  n-gram model with smoothing
  pos_tagger.py              NLTK and spaCy tagging
  parser.py                  dependency analysis and agreement rule
  cky.py                     CFG + CKY demonstration
  pipeline.py                combines all NLP stages into one report
  llm.py                     prompt filling, API call, validation, fallback, cache
tests/                       automated tests (pytest)
manual_tests/                scripts that print results for a human to read
scripts/check_llm_connection.py  checks the API key and lists available models
data/test_sentences.csv      test set with expected answers
docs/evaluation_notes.md     recorded evaluation results and observations
```