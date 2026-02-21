"""
lexeme_parser.py
================
Parses Duolingo lexeme strings from a small dataset sample into structured
grammatical categories. Run standalone or import into the notebook.

Lexeme string format:  surface_form/lemma<tag1><tag2>...
Example:  lernt/lernen<vblex><pri><p3><sg>
"""

import re
import pandas as pd
import numpy as np
from collections import Counter

# ---------------------------------------------------------------------------
# Tag taxonomy  (compiled from Apertium morphological analyser conventions)
# ---------------------------------------------------------------------------

# Part of Speech (POS) — the FIRST tag inside <...> usually
POS_MAP = {
    # Verbs
    "vblex": "verb_lexical",
    "vbser": "verb_ser",          # to be (ser/être/sein)
    "vbhaver": "verb_haver",      # to have (auxiliary)
    "vbmod": "verb_modal",
    "vaux": "verb_auxiliary",
    "vbdo": "verb_do",
    # Nouns
    "n": "noun",
    "np": "proper_noun",
    "abbr": "abbreviation",
    # Determiners / Articles
    "det": "determiner",
    # Adjectives
    "adj": "adjective",
    # Adverbs
    "adv": "adverb",
    # Pronouns
    "prn": "pronoun",
    # Prepositions
    "pr": "preposition",
    "prep": "preposition",
    # Conjunctions
    "cnj": "conjunction",
    "cnjadv": "conjunction_adverbial",
    "cnjcoo": "conjunction_coordinating",
    "cnjsub": "conjunction_subordinating",
    # Interjections
    "ij": "interjection",
    # Numbers
    "num": "numeral",
    "ord": "ordinal",
    # Other
    "sym": "symbol",
    "pun": "punctuation",
    "x": "other",
}

# Tense / Mood (for verbs)
TENSE_MAP = {
    "inf":  "infinitive",
    "ger":  "gerund",
    "pp":   "past_participle",
    "pri":  "present_indicative",
    "pii":  "past_imperfect_indicative",
    "pis":  "preterite",
    "prs":  "present_subjunctive",
    "pis":  "imperfect_subjunctive",
    "imp":  "imperative",
    "cni":  "conditional",
    "fti":  "future_indicative",
    "fts":  "future_subjunctive",
    "pmp":  "pluperfect",
    "pprs": "present_participle",
    "ppa":  "past_anterior",
    # Additional tense tags found in the data
    "pres": "present",
    "pst":  "past",
    "past": "past",
    "ifi":  "past_simple",
    "ifi":  "past_simple",
}

# Person
PERSON_MAP = {
    "p1": "1st_person",
    "p2": "2nd_person",
    "p3": "3rd_person",
}

# Number
NUMBER_MAP = {
    "sg": "singular",
    "pl": "plural",
    "sp": "singular_or_plural",   # ambiguous
}

# Grammatical Gender
GENDER_MAP = {
    "m":  "masculine",
    "f":  "feminine",
    "nt": "neuter",
    "mf": "masculine_or_feminine",
    "GD": "any_gender",
}

# Grammatical Case
CASE_MAP = {
    "nom": "nominative",
    "acc": "accusative",
    "dat": "dative",
    "gen": "genitive",
    "voc": "vocative",
    "ins": "instrumental",
    "loc": "locative",
    "abl": "ablative",
}

# Definiteness (determiners / nouns in some languages)
DEF_MAP = {
    "def": "definite",
    "ind": "indefinite",
    "dem": "demonstrative",
    "pos": "possessive",
    "itg": "interrogative",
    "rel": "relative",
    "qnt": "quantifier",
}

# Adjective degree
DEGREE_MAP = {
    "comp": "comparative",
    "sup":  "superlative",
    "sint": "synthetic_superlative",
}

# Pronoun sub-type
PRONOUN_MAP = {
    "ref": "reflexive",
    "obj": "object",
    "subj": "subject",
    "dem": "demonstrative",
    "itg": "interrogative",
    "rel": "relative",
    "excl": "exclamative",
    "ind": "indefinite",
    "pers": "personal",
}

# Verb sub-type extras
VERB_EXTRA = {
    "actv": "active_voice",
    "pasv": "passive_voice",
    "trans": "transitive",
    "intrans": "intransitive",
}

# Conjunction sub-type
CONJ_MAP = {
    "coo": "coordinating",
    "sub": "subordinating",
}

# Apertium missing-feature markers (tag present but value unspecified for this form)
APERTIUM_META = {
    "*numb", "*sf", "*case", "*pers", "*gndr",
    "tn",        # invariant / no tone
    "apos",      # apostrophe elision form
    "sw",        # spelling variant
    "suff",      # suffix
    "an",        # animate
    "mix",       # mixed/ambiguous
    "preadv",    # pre-adverb
    "attr",      # attributive use
    "pred",      # predicative use
    "pro",       # pronoun attribute (used inside det)
    "pron",      # pronoun (alternate tag)
}

# Multi-word expression / construction labels (start with '@')
MWE_PREFIX = "@"


# ---------------------------------------------------------------------------
# Core parsing function
# ---------------------------------------------------------------------------

def parse_lexeme(lexeme_str: str) -> dict:
    """
    Parse a single lexeme_string into a structured dict.

    Returns
    -------
    dict with keys:
      surface_form, lemma, raw_tags, pos, pos_label,
      tense, person, number, gender, case, definiteness,
      degree, pronoun_type, verb_voice, conjunction_type,
      unknown_tags
    """
    if not isinstance(lexeme_str, str):
        return {}

    # 1. Split surface_form/lemma from tags
    #    Pattern: everything before the first '<' is  surface/lemma
    tag_match = re.search(r'<', lexeme_str)
    if tag_match:
        word_part = lexeme_str[:tag_match.start()]
        tag_part  = lexeme_str[tag_match.start():]
    else:
        word_part = lexeme_str
        tag_part  = ""

    # 2. Extract surface form and lemma
    if "/" in word_part:
        surface_form, lemma = word_part.split("/", 1)
    else:
        surface_form = word_part
        lemma = word_part

    # 3. Extract all tags in order
    raw_tags = re.findall(r'<([^>]+)>', tag_part)

    # 4. Map each tag to its grammatical category
    pos           = None
    pos_label     = None
    tense         = None
    person        = None
    number        = None
    gender        = None
    case          = None
    definiteness  = None
    degree        = None
    pronoun_type  = None
    verb_voice    = None
    conj_type     = None
    unknown_tags  = []

    for tag in raw_tags:
        t = tag.lower()

        if t in POS_MAP and pos is None:
            pos       = tag
            pos_label = POS_MAP[t]

        elif t in TENSE_MAP:
            tense = TENSE_MAP[t]

        elif t in PERSON_MAP:
            person = PERSON_MAP[t]

        elif t in NUMBER_MAP:
            number = NUMBER_MAP[t]

        elif t in GENDER_MAP:
            gender = GENDER_MAP[t]

        elif t in CASE_MAP:
            case = CASE_MAP[t]

        elif t in DEF_MAP:
            definiteness = DEF_MAP[t]

        elif t in DEGREE_MAP:
            degree = DEGREE_MAP[t]

        elif t in PRONOUN_MAP:
            pronoun_type = PRONOUN_MAP[t]

        elif t in VERB_EXTRA:
            verb_voice = VERB_EXTRA[t]

        elif t in CONJ_MAP:
            conj_type = CONJ_MAP[t]

        # Silently absorb Apertium meta-markers and MWE construction labels
        elif t in APERTIUM_META or tag.startswith(MWE_PREFIX):
            pass  # known, deliberately ignored

        # Secondary POS tag (e.g. "det" inside a determiner sequence)
        elif t in POS_MAP:
            pass  # already have pos from first tag

        else:
            unknown_tags.append(tag)

    return {
        "surface_form":    surface_form,
        "lemma":           lemma,
        "raw_tags":        raw_tags,
        "pos":             pos,
        "pos_label":       pos_label,
        "tense":           tense,
        "person":          person,
        "number":          number,
        "gender":          gender,
        "case":            case,
        "definiteness":    definiteness,
        "degree":          degree,
        "pronoun_type":    pronoun_type,
        "verb_voice":      verb_voice,
        "conjunction_type": conj_type,
        "unknown_tags":    unknown_tags,
    }


# ---------------------------------------------------------------------------
# Apply to a DataFrame
# ---------------------------------------------------------------------------

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with a 'lexeme_string' column,
    parse each row and add all grammatical columns.
    """
    parsed = df["lexeme_string"].apply(parse_lexeme).apply(pd.Series)
    # Drop raw_tags and unknown_tags from direct concat (keep as lists)
    scalar_cols = [c for c in parsed.columns if c not in ("raw_tags", "unknown_tags")]
    return pd.concat([df, parsed[scalar_cols]], axis=1)


# ---------------------------------------------------------------------------
# Main: load small sample, parse, and show summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_ROWS = 3000

    print(f"Loading first {SAMPLE_ROWS} rows from learning_traces.13m.csv ...")
    df = pd.read_csv(
        "learning_traces.13m.csv",
        nrows=SAMPLE_ROWS,
        parse_dates=False,
    )
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    print()

    print("Parsing lexeme strings ...")
    df_enriched = enrich_dataframe(df)
    print(f"  New columns added: {[c for c in df_enriched.columns if c not in df.columns]}")
    print()

    # -----------------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------------

    print("=" * 60)
    print("PART-OF-SPEECH DISTRIBUTION")
    print("=" * 60)
    pos_counts = df_enriched["pos_label"].value_counts(dropna=False)
    print(pos_counts.to_string())
    print()

    print("=" * 60)
    print("POS → SUBCATEGORY BREAKDOWN")
    print("=" * 60)

    POS_GROUPS = {
        "verb_lexical":  ["tense", "person", "number"],
        "noun":          ["gender", "number", "case"],
        "determiner":    ["definiteness", "gender", "number", "case"],
        "adjective":     ["gender", "number", "case", "degree"],
        "pronoun":       ["pronoun_type", "person", "number", "gender", "case"],
        "adverb":        ["degree"],
        "conjunction":   ["conjunction_type"],
    }

    for pos_label, subcats in POS_GROUPS.items():
        subset = df_enriched[df_enriched["pos_label"] == pos_label]
        if subset.empty:
            continue
        print(f"\n--- {pos_label.upper()}  (n={len(subset)}) ---")
        for cat in subcats:
            if cat in subset.columns:
                vc = subset[cat].value_counts(dropna=True)
                if vc.empty:
                    print(f"  {cat}: (no data)")
                else:
                    vals = ", ".join(f"{k}: {v}" for k, v in vc.items())
                    print(f"  {cat}: {vals}")

    print()
    print("=" * 60)
    print("TENSE / MOOD DISTRIBUTION (verbs only)")
    print("=" * 60)
    verb_mask = df_enriched["pos_label"].str.startswith("verb", na=False)
    verb_tense = df_enriched.loc[verb_mask, "tense"].value_counts(dropna=True)
    print(verb_tense.to_string() if not verb_tense.empty else "(none)")

    print()
    print("=" * 60)
    print("GENDER DISTRIBUTION (nouns + adjectives)")
    print("=" * 60)
    gender_mask = df_enriched["pos_label"].isin(["noun", "adjective", "proper_noun"])
    gender_counts = df_enriched.loc[gender_mask, "gender"].value_counts(dropna=True)
    print(gender_counts.to_string() if not gender_counts.empty else "(none)")

    print()
    print("=" * 60)
    print("CASE DISTRIBUTION")
    print("=" * 60)
    case_counts = df_enriched["case"].value_counts(dropna=True)
    print(case_counts.to_string() if not case_counts.empty else "(none)")

    print()
    print("=" * 60)
    print("SAMPLE PARSED ROWS (first 15, key columns)")
    print("=" * 60)
    show_cols = ["lexeme_string", "pos_label", "tense", "person", "number", "gender", "case", "definiteness"]
    available = [c for c in show_cols if c in df_enriched.columns]
    print(df_enriched[available].head(15).to_string())

    print()
    print("Done. df_enriched has", df_enriched.shape[1], "columns.")
