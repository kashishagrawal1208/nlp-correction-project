"""
app.py
Phase 11: Streamlit user interface.

Run from the project root with:  streamlit run app.py
This file only DISPLAYS results. All NLP work is done in src/pipeline.py
and src/llm.py.
"""

import streamlit as st

from src.llm import correct_text
from src.pipeline import run_pipeline

EXAMPLES = [
    "",
    "I recieved teh letter yesterday. She go to school evry day.",
    "I want to by a knew car.",
    "The boys in the red car goes home.",
    "Yesterday I go to the market and buyed some apple.",
    "The boy eats red apples.",
]

st.set_page_config(page_title="Hybrid NLP-LLM Correction", layout="wide")


# ---------- Small helper: fill the text box from the example list ----------

def use_example():
    """Copy the chosen example sentence into the text box."""
    st.session_state.user_text = st.session_state.example_choice


st.session_state.setdefault("user_text", "")


# ---------- Header and input ----------

st.title("A Hybrid NLP-LLM Framework for Context-Aware Spelling and Grammar Correction")
st.caption(
    "Traditional NLP analyses the text first (normalization, tokenization, spelling, "
    "n-grams, POS tagging, dependency parsing). An LLM then makes the final correction "
    "using that analysis as evidence."
)

st.selectbox("Load an example (optional)", EXAMPLES, key="example_choice", on_change=use_example)

st.text_area(
    "Input text",
    key="user_text",
    height=150,
    placeholder="Enter your English text here...",
)

analyze_clicked = st.button("Analyze Text", type="primary")


# ---------- Run everything when the button is pressed ----------

if analyze_clicked:
    text = st.session_state.user_text.strip()

    if not text:
        st.warning("Please enter some text first.")
        st.stop()

    with st.spinner("Running NLP pipeline (the first run loads corpora and models, so it can take up to a minute)..."):
        try:
            report = run_pipeline(text)
        except Exception as error:
            st.error(f"The NLP pipeline failed: {error}")
            st.stop()

    with st.spinner("Asking the LLM for a context-aware correction..."):
        result = correct_text(report)

    # ---- Section 1: Normalized text ----
    st.header("1. Normalized Text")
    st.code(report["normalized_text"], language=None)

    # ---- Section 2: Tokens ----
    st.header("2. Tokens")
    for number, item in enumerate(report["tokenized"], start=1):
        st.markdown(f"**Sentence {number}:** {item['sentence']}")
        st.write("Tokens (with punctuation):", item["tokens"])
        st.write("Words (punctuation removed):", item["words"])

    # ---- Section 3: Spelling errors ----
    st.header("3. Spelling Errors")
    if not report["spelling_errors"]:
        st.info("No non-word spelling errors detected. (Real-word errors such as 'by' for "
                "'buy' cannot be found by a dictionary; the n-gram section and the LLM handle those.)")
    for error in report["spelling_errors"]:
        best = error["best_in_context"] or "no candidate found"
        st.markdown(f"**{error['word']}** → best fit in context: **{best}** "
                    f"(sentence {error['sentence'] + 1}, between "
                    f"'{error['left_word']}' and '{error['right_word']}')")
        if error["candidates"]:
            st.dataframe(
                [{"Candidate": c["word"],
                  "Edit distance": c["distance"],
                  "Context score (closer to 0 = better fit)": c["context_score"]}
                 for c in error["candidates"]],
                hide_index=True,
            )

    # ---- Section 4: N-gram analysis ----
    st.header("4. N-Gram Analysis")
    for number, ngram in enumerate(report["ngram_analysis"], start=1):
        st.markdown(f"**Sentence {number}**")
        left, right = st.columns(2)
        left.metric("Average log-probability per word (closer to 0 = more typical)",
                    ngram["avg_log_prob_per_word"])
        right.metric("Word pairs never seen in the corpus", len(ngram["unseen_bigrams"]))
        if ngram["unseen_bigrams"]:
            st.write("Unseen bigrams:", ngram["unseen_bigrams"])
    st.caption("Smoothed bigram model (add-one) trained on the Brown corpus. "
               "The corpus is small, so an unseen pair is a hint, not proof, of an error.")

    # ---- Section 5: POS tags ----
    st.header("5. POS Tags")
    for number, tagged in enumerate(report["pos_tags"], start=1):
        st.markdown(f"**Sentence {number}:** {tagged['sentence']}")
        st.dataframe(
            [{"Word": t["word"], "Universal tag": t["upos"], "Penn Treebank tag": t["ptb"]}
             for t in tagged["spacy"]],
            hide_index=True,
        )
        if tagged["disagreements"]:
            st.write("NLTK and spaCy disagree on:", tagged["disagreements"])

    # ---- Section 6: Grammar / dependency analysis ----
    st.header("6. Grammar / Dependency Analysis")
    for number, grammar in enumerate(report["grammar_analysis"], start=1):
        st.markdown(f"**Sentence {number}:** {grammar['sentence']}")
        st.text("Dependency tree:\n" + grammar["tree"])
        st.write("Noun phrases:", grammar["noun_phrases"])
        st.write("Verb structure:", grammar["clauses"])
        if grammar["possible_issues"]:
            for issue in grammar["possible_issues"]:
                st.warning(issue)
        else:
            st.success("No rule-based grammar warnings.")

    # ---- Section 7: LLM correction ----
    st.header("7. LLM Correction")
    if result["source"] == "fallback":
        st.warning("The LLM could not be used, so this is a rule-based FALLBACK "
                   "(spelling only). Reason: " + str(result["error"]))

    st.markdown("**Original**")
    st.code(report["original_text"], language=None)
    st.markdown("**Corrected**")
    st.code(result["corrected_text"], language=None)

    st.markdown("**Explanation**")
    st.write(result["summary"])
    if result["changes"]:
        st.dataframe(
            [{"Original": c.get("original", ""),
              "Corrected": c.get("corrected", ""),
              "Type": c.get("type", ""),
              "Explanation": c.get("explanation", "")}
             for c in result["changes"]],
            hide_index=True,
        )
    else:
        st.info("No changes were made.")

    with st.expander("Show the exact prompt sent to the LLM"):
        st.text(result["prompt"])