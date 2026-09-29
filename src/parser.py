"""
parser.py
Phase 7: Dependency (grammar) analysis with spaCy.

Dependency grammar: every word is linked to one "head" word with a labelled
relation (subject, object, modifier ...). We extract these links, find the
subject-verb-object structure, and run a simple agreement check.
"""

from src.pos_tagger import get_spacy_model

SUBJECT_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "obj", "iobj", "dative", "attr"}
MODIFIER_DEPS = {"amod", "advmod", "nummod", "compound", "poss"}

SINGULAR_PRONOUNS = {"he", "she", "it"}
PLURAL_PRONOUNS = {"they", "we", "you", "i"}


def dependency_table(doc):
    """One row per token: the word, its relation, and the word it depends on."""
    return [
        {
            "word": token.text,
            "relation": token.dep_,
            "head": token.head.text,
            "pos": token.pos_,
        }
        for token in doc
    ]


def tree_lines(token, depth=0):
    """Draw the dependency tree as indented text, starting from one token."""
    lines = ["    " * depth + f"{token.text} ({token.dep_})"]
    for child in token.children:
        lines.extend(tree_lines(child, depth + 1))
    return lines


def extract_clauses(doc):
    """Find each verb and its subject / object / modifiers.

    Returns e.g. [{"verb": "eats", "subjects": ["boy"],
                   "objects": ["apples"], "modifiers": ["quickly"]}]"""
    clauses = []
    for token in doc:
        if token.pos_ not in ("VERB", "AUX") and token.dep_ != "ROOT":
            continue
        subjects = [c.text for c in token.children if c.dep_ in SUBJECT_DEPS]
        objects = [c.text for c in token.children if c.dep_ in OBJECT_DEPS]
        modifiers = [c.text for c in token.children if c.dep_ in MODIFIER_DEPS]
        # Also collect objects of prepositions: "to (school)"
        for child in token.children:
            if child.dep_ == "prep":
                objects.extend(g.text for g in child.children if g.dep_ == "pobj")
        # Skip helper verbs (like "is" in "is going") that have no arguments
        if subjects or objects or token.dep_ == "ROOT":
            clauses.append({
                "verb": token.text,
                "subjects": subjects,
                "objects": objects,
                "modifiers": modifiers,
            })
    return clauses


def check_agreement(doc):
    """Simple RULE-BASED subject-verb agreement check.

    Uses the dependency link (subject -> verb) plus Penn Treebank tags:
      VBZ = present, 3rd person singular ("goes")
      VBP = present, other persons ("go")
    The verb form that shows agreement can be the main verb ("She go")
    or an auxiliary attached to it ("The boys is playing", "He don't like").
    This is a heuristic: it only catches present-tense mismatches."""
    issues = []
    for token in doc:
        if token.dep_ != "nsubj":
            continue
        subject = token.text.lower()
        is_singular = token.tag_ in ("NN", "NNP") or subject in SINGULAR_PRONOUNS
        is_plural = token.tag_ in ("NNS", "NNPS") or subject in PLURAL_PRONOUNS

        # The subject's head verb, plus any auxiliaries attached to that verb
        verbs = [token.head] + [c for c in token.head.children if c.dep_ == "aux"]
        for verb in verbs:
            if verb.tag_ not in ("VBP", "VBZ"):
                continue
            if is_singular and verb.tag_ == "VBP":
                issues.append(f"Possible agreement error: singular subject "
                              f"'{token.text}' with verb '{verb.text}' (VBP)")
            elif is_plural and verb.tag_ == "VBZ":
                issues.append(f"Possible agreement error: plural subject "
                              f"'{token.text}' with verb '{verb.text}' (VBZ)")

    if not any(t.pos_ in ("VERB", "AUX") for t in doc):
        issues.append("No verb found: the sentence may be a fragment")
    return issues


def analyze_sentence(sentence):
    """Full grammar report for ONE sentence."""
    doc = get_spacy_model()(sentence)
    roots = [t for t in doc if t.dep_ == "ROOT"]
    return {
        "sentence": sentence,
        "dependencies": dependency_table(doc),
        "tree": "\n".join(line for root in roots for line in tree_lines(root)),
        "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
        "clauses": extract_clauses(doc),
        "possible_issues": check_agreement(doc),
    }


def analyze_text(tokenized):
    """Grammar report for every sentence from tokenizer.tokenize_text()."""
    return [analyze_sentence(item["sentence"]) for item in tokenized]