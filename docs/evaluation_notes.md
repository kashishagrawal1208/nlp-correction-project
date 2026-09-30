# Evaluation notes (real results only)

Test set: `data/test_sentences.csv`, 25 sentences written by the author.
All numbers below describe THIS test set, not English in general.

## Spelling detection (non-word errors), `tests/eval_spelling.py`
- 12 gold misspellings: 12 true positives, 0 false positives, 0 false negatives
- Precision 1.00, recall 1.00, F1 1.00
- Correct word in the top 3 candidates: 12/12
- Correct word ranked first, edit distance only: 11/12
- Correct word ranked first, with N-gram context: 12/12
- The only difference was `evry`: `very` and `every` are both one edit away,
  so edit distance tied and word frequency picked `very`; context picked `every`.
  (One case only, not strong evidence.)

## Spelling weaknesses found on harder text (`tests/inspect_spelling.py`, not in the test set)
- False positives: `Priya` (name at sentence start), `app`, `iPhone`, `lol`
  (the lexicon comes from Brown, 1961, and NLTK's word list, so modern words are missing)
- `3pm` was not flagged (tokens that are not purely letters are skipped by design)
- Typos such as `Thier`, `exicted`, `totaly`, `awsome`, `pakage`, `hapy` were corrected well

## Agreement check (rule-based), `tests/eval_grammar.py`
- Gold agreement errors: rows 10, 11, 12, 13, 23, 24, 25 (decided before running)
- BEFORE extending the rule: 5/7 detected (recall 0.71), 0 false alarms
- Missed: row 12 "My brother don't like coffee." and row 24 "The childrens is playing in the gardan."
- Cause: the subject attaches to the main verb (VB / VBG), while the agreeing
  form is an auxiliary (`do` VBP, `is` VBZ), which the rule did not check.
- AFTER extending the rule to check auxiliaries: 7/7 detected (recall 1.00), 0 false alarms
- The rule was changed after seeing these failures on the same test set,
  so the "after" figure is optimistic.
- Sanity check on 6 correct sentences with auxiliaries: no warnings (small sample).
- Tense errors (rows 14-16) are outside this rule and are left to the LLM.

## Other observations
- The dependency parser was confused by a misspelling: in "She go to school evry day."
  it attached `school` and `evry` to `day` as compounds.
- NLTK and spaCy disagree on POS tags for misspelled words (`teh`, `evry`).
- N-gram "unseen bigrams" flag some correct pairs (`letter yesterday`), and missed
  `to by` in "I want to by a knew car." (Brown is small).
- LLM API failures met during development: 503 (server busy) and 429 (daily free
  quota); both were handled by the fallback.

  ## LLM correction, `manual_tests/eval_llm.py`
- Model: gemini-3.1-flash-lite, temperature 0.2, prompt `prompts/correction_prompt.txt`,
  a single run over all 25 test sentences (run on 30 September 2026).
- Exact match with the expected sentence: 25/25
  (agreement 4/4, article 3/3, correct 4/4, multiple 3/3, real_word 3/3, spelling 5/5, tense 3/3).
- Over-correction: 0 of 4 already-correct sentences were changed.
- Model history: `gemini-3.7-flash` was used during development, but its free daily quota
  (about 20 requests per day) was too small for the full test set, so the evaluation
  was run with `gemini-3.1-flash-lite`. The model was chosen for its quota, before seeing any results,
  and was not changed afterwards. The prompt was not changed during the evaluation.
- Limits of this result: the test set is small (25 sentences), written by the author,
  with the expected answers fixed by the author before the run; exact match is strict and
  was not confirmed on unseen data; only one run was made, and LLM output can vary
  between runs. We did NOT test the LLM without the NLP evidence, so this result does not
  show that the NLP evidence improves the correction.
- Network errors (DNS lookup failure) hit row 13 during the first run; the row was
  completed in a second run and the other rows were read from the cache.
- Row 24 was answered by an API call in two separate runs, although a cached answer was
  expected the second time; the cause is unknown. The answer matched in both runs.