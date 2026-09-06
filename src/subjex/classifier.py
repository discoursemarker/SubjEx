"""Extract and classify grammatical subject spans in English text.

The classifier uses a spaCy English transformer pipeline and returns one
tab-separated row for each detected subject. The public entry point is
``classify_subjects``.
"""

from __future__ import annotations

from typing import List, Set

import spacy
from spacy.language import Language

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
REL_PRONOUNS: Set[str] = {"who", "whom", "whose", "which", "that", "where", "when", "why", "how"}
ARTICLES: Set[str] = {"a", "an", "the"}
SUBJECT_DEPS: Set[str] = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
EXCLAMATIONS: Set[str] = {"yes", "no", "yeah", "nah", "yep", "nope"}
EXISTENTIAL_VERBS: Set[str] = {"be", "seem", "appear", "remain", "exist", "stand", "lie"}
WH_CLAUSE_LABELS = {
    "what": "what_clause",
    "whether": "whether_clause",
    "whichever": "whichever_clause",
    "whatever": "whatever_clause",
    "whoever": "whoever_clause",
    "whomever": "whomever_clause",
    "that": "that_clause",
    "where": "where_clause",
    "when": "when_clause",
    "why": "why_clause",
    "how": "how_clause",
}
DEFAULT_MODEL = "en_core_web_trf"

_pipeline: Language | None = None
_pipeline_name: str | None = None


def get_pipeline(model_name: str = DEFAULT_MODEL) -> Language:
    """Load and cache the spaCy pipeline used by the classifier."""

    global _pipeline, _pipeline_name
    if _pipeline is None or _pipeline_name != model_name:
        _pipeline = spacy.load(model_name)
        _pipeline_name = model_name
    return _pipeline

# --------------------------------------------------------------------------- #
# Span utilities                                                              #
# --------------------------------------------------------------------------- #
def add_span(unique: List[spacy.tokens.Span], new: spacy.tokens.Span) -> None:
    """Append *new* to *unique* if it is not subsumed by an existing span.

    If *new* subsumes an existing span, replace the existing span to avoid
    overlaps.  The list is modified in‑place.
    """

    # Skip if an existing span already contains *new*
    for span in unique:
        if span.start <= new.start and span.end >= new.end:
            return

    # Remove spans that are fully contained in *new*
    unique[:] = [
        span for span in unique if not (new.start <= span.start and new.end >= span.end)
    ]
    unique.append(new)


def is_existential_there(sent: spacy.tokens.Span) -> bool:
    """Return *True* when the sentence starts with existential **there**."""

    return sent[0].lower_ == "there" and sent.root.lemma_ in EXISTENTIAL_VERBS


def is_sentence_initial_infinitive(sent: spacy.tokens.Span) -> bool:
    """Detect an initial bare infinitive subject (``to + VERB``)."""

    return len(sent) >= 2 and sent[0].lemma_ == "to" and sent[1].pos_ == "VERB"

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def classify_subjects(text: str, pipeline: Language | None = None) -> str:
    """Extract and label subjects in each declarative sentence in ``text``.

    Each output row has four tab-separated fields: sentence, subject span,
    non-punctuation token count, and comma-separated labels. Questions and
    sentences without a detected subject do not produce rows.
    """

    text = text.replace("\n", " ").replace("\t", " ").replace("  ", " ")
    doc = (pipeline or get_pipeline())(text)
    results: List[str] = []

    for sent in doc.sents:
        if sent.text.strip().endswith("?"):
            # Questions are outside the scope of this classifier.
            continue

        subjects: List[spacy.tokens.Span] = []

        # ------------------------------------------------------------------ #
        # 1) Canonical subjects (including coordination)                     #
        # ------------------------------------------------------------------ #
        for tok in sent:
            if tok.dep_ not in SUBJECT_DEPS:
                continue

            # Ignore subjects embedded in dependent clauses and complements.
            if tok.head.dep_ in {"xcomp", "ccomp", "acl", "relcl", "advcl", "oprd", "dobj"}:
                continue

            # Keep subjects that ultimately depend on the sentence predicate.
            if not any(anc.dep_ in {"ROOT", "aux", "auxpass"} for anc in tok.ancestors):
                continue

            span = doc[tok.left_edge.i : tok.right_edge.i + 1]
            if len(span) == 1 and span[0].lower_ in REL_PRONOUNS:
                continue  # Ignore a relative pronoun without its antecedent.

            add_span(subjects, span)

            # Include coordinated subjects such as "A and B".
            for conj in tok.conjuncts:
                cspan = doc[conj.left_edge.i : conj.right_edge.i + 1]
                if len(cspan) == 1 and cspan[0].lower_ in REL_PRONOUNS:
                    continue
                add_span(subjects, cspan)

        # ------------------------------------------------------------------ #
        # 2) Existential "there"                                            #
        # ------------------------------------------------------------------ #
        if is_existential_there(sent):
            add_span(subjects, doc[sent[0].i : sent[0].i + 1])

        # ------------------------------------------------------------------ #
        # 3) Sentence‑initial infinitive                                    #
        # ------------------------------------------------------------------ #
        if is_sentence_initial_infinitive(sent):
            span = doc[sent[0].left_edge.i : sent[1].right_edge.i + 1]
            if span.root.dep_ in SUBJECT_DEPS:
                add_span(subjects, span)

        if not subjects:
            continue

        # ------------------------------------------------------------------ #
        # 4) Label each subject span                                        #
        # ------------------------------------------------------------------ #
        for span in subjects:
            txt = span.text
            ln = sum(1 for t in span if not t.is_punct)
            head = span.root
            labels: Set[str] = set()

            # Coordination ("A and B")
            if any(t.dep_ == "cc" and t.lower_ == "and" for t in span) and any(
                t.dep_ == "conj" for t in span
            ):
                labels.add("A_and_B")

            # Coordination ("A or B")
            if any(t.dep_ == "cc" and t.lower_ == "or" for t in span) and any(
                t.dep_ == "conj" for t in span
            ):
                labels.add("A_or_B")

            # Label clause-like subjects from their opening word.
            first = span[0].lower_
            if first in WH_CLAUSE_LABELS:
                labels.add(WH_CLAUSE_LABELS[first])

            # Numeral subjects
            if span[0].pos_ == "NUM" or span[0].like_num:
                labels.add("Numeral")

            # Detect a "that" clause attached as an apposition or complement.
            if "that" in [t.lower_ for t in span] and any(
                ch.dep_ in {"appos", "ccomp"} for ch in head.children
            ):
                labels.add("that_appos")

            # Label common pre-head modifiers.
            if any(ch.dep_ == "compound" for ch in head.lefts):
                labels.add("Compound")
            if any(ch.dep_ == "amod" for ch in head.lefts):
                labels.add("Adjective")

            # Label definite and indefinite articles.
            for det in span:
                if det.dep_ == "det" and det.tag_ == "DT" and det.lemma_ in ARTICLES:
                    if det.lemma_ == "the":
                        labels.add("Def_article")
                    else:
                        labels.add("Indef_article")

            # Post‑head modifiers
            for ch in head.rights:
                if ch.dep_ == "prep":
                    labels.add(f"Prep_{ch.text.lower()}")
                elif ch.dep_ == "xcomp" and ch.head.dep_ == "acl":
                    labels.add("Infinitive")

            # Distinguish gerunds from participles using their syntactic role.
            for tok in span:
                if tok.tag_ == "VBG":
                    if tok == head or tok.dep_ in {"aux", "auxpass", "mark"}:
                        labels.add("Gerund")
                    else:
                        labels.add("PresentParticiple")

            # Infinitive marker "to"
            if any(t.lemma_ == "to" and t.pos_ == "PART" for t in span):
                labels.add("Infinitive")

            # Possessive
            for t in span:
                if (t.dep_ == "poss" and t.head == head) or (
                    t.dep_ == "nsubj" and head.tag_ == "VBG" and t.head == head
                ):
                    labels.add("Possessive")
                    break

            # Preserve the legacy label for two-token article+noun spans.
            if (
                any(
                    t.dep_ == "det" and t.tag_ == "DT" and t.lemma_ in ARTICLES for t in span
                )
                and ln == 2
            ):
                labels.add("Article_only")

            # Label non-article determiner-initial spans.
            if span[0].dep_ == "det" and span[0].tag_ == "DT" and span[0].lemma_ not in ARTICLES:
                labels.add("Determiner")

            # Label simple one-word nominal subjects.
            if ln == 1 and any(t.tag_ == "NNS" for t in span):
                labels.add("Plural_only")
            if ln == 1 and any(t.pos_ == "PROPN" for t in span):
                labels.add("Proper_Noun")

            # Apply specialized pronoun labels.
            lower_txt = txt.lower()
            if lower_txt == "it":
                labels.add(_classify_it(sent))
            elif lower_txt == "there":
                labels.add("There")
            elif head.pos_ == "PRON":
                labels.add("Pronoun")

            if (
                ln == 1
                and head.tag_ in {"NN", "NNP"}
                and "Plural_only" not in labels
                and "Proper_Noun" not in labels
                and "Pronoun" not in labels
            ):
                labels.add("Singular_only")

            # Relative pronouns inside span
            if any(t.lower_ in REL_PRONOUNS and t.pos_ != "SCONJ" for t in span):
                labels.add("Rel")

            # Prepositions inside span
            for p in [t.lower_ for t in span if t.pos_ == "ADP"]:
                labels.add(f"Prep_{p}")

            # Label subject spans beginning with "such".
            if span[0].lower_ == "such":
                labels.add("such_phrase")

            # Label one-word exclamations used as quoted or metalinguistic subjects.
            if ln == 1 and span[0].lower_ in EXCLAMATIONS:
                labels.add("Exclamation")

            # Quoted subjects
            if any(tok.is_quote for tok in span):
                labels.add("Quote")

            # Detect relative clauses without an explicit relative pronoun.
            if any(tok.dep_ == "relcl" for tok in span) and not any(
                tok.lower_ in REL_PRONOUNS for tok in span
            ):
                labels.add("ContactClause")

            results.append(
                f"{sent.text.strip()}\t{txt}\t{ln}\t{', '.join(sorted(labels))}"
            )

    return "\n".join(results)

# --------------------------------------------------------------------------- #
# Fine‑grained "it_*" classification                                          #
# --------------------------------------------------------------------------- #

def _classify_it(sent: spacy.tokens.Span) -> str:
    """Return a specialized label for the pronoun ``it``."""

    root = sent.root

    if root.lemma_ == "be":
        noun_comp = any(
            ch.dep_ in {"attr", "nsubj", "dobj", "pobj"} and ch.pos_ in {"NOUN", "PROPN"}
            for ch in root.children
        )
        has_rel = any(t.lower_ in REL_PRONOUNS for t in sent)
        if noun_comp and has_rel:
            return "it_emph"

    if root.lemma_ in {"seem", "appear"} and any(
        ch.dep_ == "ccomp" and any(t.lower_ == "that" for t in ch.subtree) for ch in root.children
    ):
        return "it_seems"

    has_adj_comp = any(ch.dep_ == "acomp" and ch.pos_ == "ADJ" for ch in root.children)
    has_to_inf = any(any(t.lemma_ == "to" for t in ch.subtree) for ch in root.children)
    if has_adj_comp and has_to_inf:
        return "it_to"

    if any(
        ch.dep_ in {"ccomp", "xcomp"} and any(t.lower_ == "that" for t in ch.subtree) for ch in root.children
    ):
        return "it_that"

    return "Pronoun"
