"""
build_duo_data.py
=================
Reads learning_traces.13m.csv in chunks, adds grammatical columns parsed
from lexeme_string using vectorised pandas regex (no per-row Python loops),
and streams the result to duo_data.csv.

All missing/inapplicable values are left as NaN → empty in CSV output.

Columns added:
  surface_form, lemma, pos_label, tense, person, number,
  gender, case, definiteness, degree

Usage:
    cd ~/Documents/Projects/Duoling_datathon
    python3 build_duo_data.py
"""

import time
import re
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV  = "learning_traces.13m.csv"
OUTPUT_CSV = "duo_data.csv"
CHUNK_SIZE = 500_000        # rows per chunk — tune up/down as needed

# ---------------------------------------------------------------------------
# All regex patterns  (applied via str.extract on the whole column at once)
# ---------------------------------------------------------------------------

# 1.  surface_form  — everything before the first '/'
#     lemma         — between '/' and first '<'
RE_SURFACE = r'^([^/]+)'
RE_LEMMA   = r'^[^/]*/([^<]+)'

# 2.  POS — first <tag> group
#     We map raw tag → readable label using str.replace (no looping)
RE_POS_RAW = r'<(vblex|vbser|vbhaver|vbmod|vaux|vbdo|n(?!p)|np|abbr|acr|det|predet|adj|adv|prn|pprep|pr(?!ep)|prep|cnj(?:adv|coo|sub)?|ij|num|ord|sym|pun|x)>'

POS_LABELS = {
    "vblex":   "verb_lexical",
    "vbser":   "verb_ser",
    "vbhaver": "verb_haver",
    "vbmod":   "verb_modal",
    "vaux":    "verb_auxiliary",
    "vbdo":    "verb_do",
    "n":       "noun",
    "np":      "proper_noun",
    "abbr":    "abbreviation",
    "det":     "determiner",
    "adj":     "adjective",
    "adv":     "adverb",
    "prn":     "pronoun",
    "pr":      "preposition",
    "prep":    "preposition",
    "cnj":     "conjunction",
    "cnjadv":  "conjunction_adverbial",
    "cnjcoo":  "conjunction_coordinating",
    "cnjsub":  "conjunction_subordinating",
    "ij":      "interjection",
    "num":     "numeral",
    "ord":     "ordinal",
    "sym":     "symbol",
    "pun":     "punctuation",
    "x":       "other",
    "predet":  "pre_determiner",
    "acr":     "acronym",
    "pprep":   "post_preposition",
}

# 3.  Tense / mood  — first match wins
RE_TENSE = (
    r'<(inf|ger|pp|pri|pii|pis|prs|imp|cni|fti|fts|pmp|pprs|ppa|pres|pst|past|ifi)>'
)
TENSE_LABELS = {
    "inf":  "infinitive",       "ger":  "gerund",
    "pp":   "past_participle",  "pri":  "present_indicative",
    "pii":  "past_imperfect_indicative",
    "pis":  "preterite",        "prs":  "present_subjunctive",
    "imp":  "imperative",       "cni":  "conditional",
    "fti":  "future_indicative","fts":  "future_subjunctive",
    "pmp":  "pluperfect",       "pprs": "present_participle",
    "ppa":  "past_anterior",    "pres": "present",
    "pst":  "past",             "past": "past",
    "ifi":  "past_simple",
}

# 4.  Person
RE_PERSON = r'<(p1|p2|p3)>'
PERSON_LABELS = {"p1": "1st_person", "p2": "2nd_person", "p3": "3rd_person"}

# 5.  Number
RE_NUMBER = r'<(sg|pl|sp|nn)>'
NUMBER_LABELS = {"sg": "singular", "pl": "plural", "sp": "singular_or_plural", "nn": "no_number"}

# 6.  Gender
RE_GENDER = r'<(mf|nt|aa|m(?!f)|f(?!t))>'
GENDER_LABELS = {"m": "masculine", "f": "feminine", "nt": "neuter", "mf": "masculine_or_feminine", "aa": "any_gender"}

# 7.  Case
RE_CASE = r'<(nom|acc|dat|gen|voc|ins|loc|abl)>'
CASE_LABELS = {
    "nom": "nominative", "acc": "accusative", "dat": "dative",
    "gen": "genitive",   "voc": "vocative",   "ins": "instrumental",
    "loc": "locative",   "abl": "ablative",
}

# 8.  Definiteness
RE_DEF = r'<(def|ind|dem|pos|itg|rel|qnt)>'
DEF_LABELS = {
    "def": "definite",    "ind": "indefinite", "dem": "demonstrative",
    "pos": "possessive",  "itg": "interrogative", "rel": "relative",
    "qnt": "quantifier",
}

# 9.  Adjective degree
RE_DEGREE = r'<(comp|sup|sint)>'
DEGREE_LABELS = {"comp": "comparative", "sup": "superlative", "sint": "synthetic_superlative"}

# 10. Pronoun sub-type
RE_PRONOUN_TYPE = r'<(ref|obj|subj|excl|pers)>'
PRONOUN_LABELS = {
    "ref":  "reflexive",  "obj":  "object", "subj": "subject",
    "excl": "exclamative","pers": "personal",
}

# 11. Adjective declension type (German strong/weak/mixed)
RE_ADJ_DECL = r'<(st|wk|un)>'
ADJ_DECL_LABELS = {"st": "strong", "wk": "weak", "un": "uninflected"}


# ---------------------------------------------------------------------------
# Vectorised enricher
# ---------------------------------------------------------------------------

def enrich_chunk(df: pd.DataFrame) -> pd.DataFrame:
    s = df["lexeme_string"]

    out = df.copy()

    # surface_form / lemma
    out["surface_form"] = s.str.extract(RE_SURFACE, expand=False)
    out["lemma"]        = s.str.extract(RE_LEMMA,   expand=False)

    # POS
    pos_raw = s.str.extract(RE_POS_RAW, expand=False)
    out["pos_label"] = pos_raw.map(POS_LABELS)

    # Tense
    tense_raw = s.str.extract(RE_TENSE, expand=False)
    out["tense"] = tense_raw.map(TENSE_LABELS)

    # Person
    person_raw = s.str.extract(RE_PERSON, expand=False)
    out["person"] = person_raw.map(PERSON_LABELS)

    # Number
    number_raw = s.str.extract(RE_NUMBER, expand=False)
    out["number"] = number_raw.map(NUMBER_LABELS)

    # Gender
    gender_raw = s.str.extract(RE_GENDER, expand=False)
    out["gender"] = gender_raw.map(GENDER_LABELS)

    # Case
    case_raw = s.str.extract(RE_CASE, expand=False)
    out["case"] = case_raw.map(CASE_LABELS)

    # Definiteness
    def_raw = s.str.extract(RE_DEF, expand=False)
    out["definiteness"] = def_raw.map(DEF_LABELS)

    # Degree
    degree_raw = s.str.extract(RE_DEGREE, expand=False)
    out["degree"] = degree_raw.map(DEGREE_LABELS)

    # Pronoun sub-type
    pron_raw = s.str.extract(RE_PRONOUN_TYPE, expand=False)
    out["pronoun_type"] = pron_raw.map(PRONOUN_LABELS)

    # Adjective declension (strong / weak / uninflected)
    adj_decl_raw = s.str.extract(RE_ADJ_DECL, expand=False)
    out["adj_declension"] = adj_decl_raw.map(ADJ_DECL_LABELS)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    first_chunk = True
    total_rows  = 0
    chunk_num   = 0

    print(f"Reading  : {INPUT_CSV}  (chunks of {CHUNK_SIZE:,})")
    print(f"Writing  : {OUTPUT_CSV}\n")

    reader = pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk in reader:
        chunk_num  += 1
        total_rows += len(chunk)
        ct = time.time()

        enriched = enrich_chunk(chunk)

        mode   = "w" if first_chunk else "a"
        header = first_chunk
        enriched.to_csv(OUTPUT_CSV, mode=mode, header=header, index=False)
        first_chunk = False

        elapsed = time.time() - t0
        rate    = total_rows / elapsed
        print(
            f"  chunk {chunk_num:4d} | rows {total_rows:>12,} "
            f"| chunk {time.time()-ct:.1f}s "
            f"| total {elapsed:.0f}s "
            f"| ~{rate:,.0f} rows/s",
            flush=True,
        )

    print(f"\nDone! {total_rows:,} rows → '{OUTPUT_CSV}'  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
