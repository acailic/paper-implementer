"""Synthetic SLOG-like grammar for the directionality experiments.

A tiny controlled English->logical-form grammar with four templates that
mirror the two SLOG structural families relevant to the directionality claim:

  * base / intrans              : saturated clauses (sanity)
  * subj_pp / obj_pp            : MODIFIER-POSITION shift family (SLOG sec 2.2,
                                  Fig 2). Same PP, two attachment sites
                                  (subject vs object) -> tests whether the
                                  compositor resolves attachment by position.

Logical form = frozenset of (pred, subj_idx, obj_idx|None) edges. Every noun's
entity is its surface token index, so directed (CKY) and undirected (AM) systems
are compared against the same grounded gold LF.

(Recursive-depth / CP-recur family is intentionally NOT generated here: in the
paper that dimension is driven by the BERT/DeBERTa supertagger + encoder scale
-- breakdown flags H3/H4 -- not by the symbolic backend that this toy isolates.
See writeup Honest Scope.)
"""

import random

NOUNS = ["boy", "dog", "cat", "bird", "fox", "ant", "elk", "owl"]
TRANS = ["saw", "chased", "heard", "found"]   # -> Vt:
INTRANS = ["ran", "slept", "fell"]            # -> Vi:
PREPS = ["in", "near", "by"]                  # -> P:


def _cells(tokens):
    """Turn a list of lexicon token-strings into CKY cells with index entities."""
    from model import lexicon_entry
    return [lexicon_entry(tok, idx) for idx, tok in enumerate(tokens)]


def _sample_nouns(rng, k):
    return rng.sample(NOUNS, k)


def make_base(rng):
    """the N1 Vt the N2  ->  {Vt(N1,N2)}"""
    n1, n2 = _sample_nouns(rng, 2)
    vt = rng.choice(TRANS)
    tokens = ["the", f"N:{n1}", f"Vt:{vt}", "the", f"N:{n2}"]
    i_subj, i_obj = 1, 4
    gold = frozenset({(vt, i_subj, i_obj)})
    roles = {"subj": i_subj, "obj": i_obj, "verb": vt, "loc": None, "prep": None}
    return tokens, roles, gold, "base"


def make_intrans(rng):
    """the N1 Vi  ->  {Vi(N1)}"""
    n1 = rng.choice(NOUNS)
    vi = rng.choice(INTRANS)
    tokens = ["the", f"N:{n1}", f"Vi:{vi}"]
    i_subj = 1
    gold = frozenset({(vi, i_subj, None)})
    roles = {"subj": i_subj, "obj": None, "verb": vi, "loc": None, "prep": None}
    return tokens, roles, gold, "intrans"


def make_subj_pp(rng):
    """the N1 P the N3 Vt the N2  ->  {Vt(N1,N2), P(N1,N3)}   [PP modifies subj]"""
    n1, n2, n3 = _sample_nouns(rng, 3)
    vt, p = rng.choice(TRANS), rng.choice(PREPS)
    tokens = ["the", f"N:{n1}", f"P:{p}", "the", f"N:{n3}", f"Vt:{vt}", "the", f"N:{n2}"]
    i_subj, i_obj, i_loc = 1, 7, 4
    gold = frozenset({(vt, i_subj, i_obj), (p, i_subj, i_loc)})
    roles = {"subj": i_subj, "obj": i_obj, "verb": vt, "loc": i_loc, "prep": p}
    return tokens, roles, gold, "subj_pp"


def make_obj_pp(rng):
    """the N1 Vt the N2 P the N3  ->  {Vt(N1,N2), P(N2,N3)}   [PP modifies obj]"""
    n1, n2, n3 = _sample_nouns(rng, 3)
    vt, p = rng.choice(TRANS), rng.choice(PREPS)
    tokens = ["the", f"N:{n1}", f"Vt:{vt}", "the", f"N:{n2}", f"P:{p}", "the", f"N:{n3}"]
    i_subj, i_obj, i_loc = 1, 4, 7
    gold = frozenset({(vt, i_subj, i_obj), (p, i_obj, i_loc)})
    roles = {"subj": i_subj, "obj": i_obj, "verb": vt, "loc": i_loc, "prep": p}
    return tokens, roles, gold, "obj_pp"


MAKERS = {
    "base": make_base,
    "intrans": make_intrans,
    "subj_pp": make_subj_pp,
    "obj_pp": make_obj_pp,
}


def sample_set(template, n, seed=0):
    """Generate `n` instances of `template`; returns list of dicts with cells."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        tokens, roles, gold, tid = MAKERS[template](rng)
        out.append({
            "template": tid,
            "tokens": tokens,
            "cells": _cells(tokens),
            "roles": roles,
            "gold": gold,
        })
    return out


def swap_order(tokens):
    """Destructively reorder tokens to a word-order the directed system should
    REJECT (verb moved to the front). This is the word-order-sensitivity probe:
    direction must be encoded in the type for the parser to notice reordering."""
    # Find the verb token and move it to position 0.
    vidx = next(i for i, t in enumerate(tokens) if t.startswith("V"))
    moved = [tokens[vidx]] + [t for i, t in enumerate(tokens) if i != vidx]
    return moved
