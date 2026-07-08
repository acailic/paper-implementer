"""Core symbolic backends for "On the Role of Directionality in Structural
Generalization" (Wei 2026, arXiv:2607.02307).

Two neuro-symbolic compositors over the SAME lexical supertags, isolating the
single variable the paper controls (the symbolic backend, breakdown sec 5.3):

  * CCG  -- directed combinatory categorial types + deterministic CKY that uses
            **forward/backward application only** (the paper's sec 3.1 backend:
            30K-param single-linear-decoder supertags + app-only CKY).
            Slash direction encodes which side of the head an argument sits on,
            so attachment is resolved by surface position / adjacency.

  * AM   -- AM-algebra apply/modify whose operations **deliberately do not
            encode direction** (the paper's comparison system, AM-Parser). With
            no slash, attachment site is under-determined by the type system and
            must be resolved by a non-positional (role/tree-head) rule.

Everything is pure Python (no deps). Logical forms are frozensets of
predicate-argument edges (pred, subj_tok_idx, obj_tok_idx | None); because every
noun's entity is its surface token index, two LF sets are comparable by plain
equality (no graph isomorphism needed).
"""

from dataclasses import dataclass


# --------------------------------------------------------------------------
# CCG type system
# --------------------------------------------------------------------------
class Cat:
    __slots__ = ()


@dataclass(frozen=True)
class Atom(Cat):
    """A saturated atomic category: NP, N, S."""
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Fun(Cat):
    """A functor category.

    side == 'fwd'  ->  res / arg   : the argument must appear on the RIGHT.
    side == 'bwd'  ->  res \\ arg  : the argument must appear on the LEFT.
    """
    res: Cat
    arg: Cat
    side: str   # 'fwd' | 'bwd'

    def __repr__(self):
        slash = "/" if self.side == "fwd" else "\\"
        return f"({self.res!r}{slash}{self.arg!r})"


NP = Atom("NP")
N = Atom("N")
S = Atom("S")

# Derived functor categories used by the lexicon.
S_BWD_NP = Fun(S, NP, "bwd")            # intransitive VP / VP-after-object
NP_BWD_NP = Fun(NP, NP, "bwd")          # post-nominal NP modifier (PP / RC)


# --------------------------------------------------------------------------
# Cells: a category paired with a denotation.
#
# A saturated cell (category is an Atom) carries sem == ('sat', head, edges)
# where `head` is the token-index entity that "owns" the cell's meaning and
# `edges` is a frozenset of (pred, subj, obj) triples.
#
# An unsaturated cell (category is a Fun) carries sem == ('fn', k) where k
# takes the saturated sem of its argument and returns a NEW cell (cat, sem).
# Nested functors (e.g. a transitive verb res\\arg over arg) are just k returning
# another ('fn', ...) cell -- no special casing needed.
# --------------------------------------------------------------------------
def sat(head: int, edges=frozenset()):
    return ("sat", head, frozenset(edges))


def fn(k):
    return ("fn", k)


def lexicon_entry(token, head_idx):
    """Return the (cat, sem) cell for `token`, `head_idx` = token's surface index.

    Every noun occurrence gets its own entity == its token index, so the LF is
    grounded in surface position and directly comparable across systems.
    """
    if token == "the":
        # NP / N : forward, takes a common noun on the right, denotes that noun.
        return (
            Fun(NP, N, "fwd"),
            fn(lambda nsem: (NP, nsem)),
        )
    if token.startswith("N:"):           # common noun -> N
        return (N, sat(head_idx))
    if token.startswith("Vt:"):          # transitive verb: (S\NP)/NP
        pred = token.split(":", 1)[1]
        return (
            Fun(S_BWD_NP, NP, "fwd"),
            fn(lambda osem: (
                S_BWD_NP,
                fn(lambda ssem: (
                    S,
                    sat(ssem[1], ssem[2] | osem[2] | {(pred, ssem[1], osem[1])}),
                )),
            )),
        )
    if token.startswith("Vi:"):          # intransitive verb: S\NP
        pred = token.split(":", 1)[1]
        return (
            S_BWD_NP,
            fn(lambda ssem: (
                S,
                sat(ssem[1], ssem[2] | {(pred, ssem[1], None)}),
            )),
        )
    if token.startswith("P:"):           # adposition: (NP\NP)/NP
        pred = token.split(":", 1)[1]
        return (
            Fun(NP_BWD_NP, NP, "fwd"),
            fn(lambda lsem: (
                NP_BWD_NP,
                fn(lambda msem: (
                    NP,
                    sat(msem[1], msem[2] | lsem[2] | {(pred, msem[1], lsem[1])}),
                )),
            )),
        )
    raise ValueError(f"no lexicon entry for {token!r}")


# --------------------------------------------------------------------------
# Deterministic CKY over application rules only (paper sec 3.1).
# --------------------------------------------------------------------------
def _apply(fun_cell, arg_sat):
    """Apply a functor cell's k to a saturated argument sem -> new cell."""
    _, sem = fun_cell
    assert sem[0] == "fn"
    return sem[1](arg_sat)


def cky(cells, want=S):
    """App-only CKY. `cells` = list of (cat, sem).

    Returns the set of LF edge-sets (frozensets) derivable for the full span
    under category `want`. Empty if no full-span derivation exists.
    """
    n = len(cells)
    # chart[(i, j)] : dict cat -> sem   for span [i, j)
    chart = {}
    for i, cell in enumerate(cells):
        chart[(i, i + 1)] = {cell[0]: cell[1]}

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            entries = {}
            for k in range(i + 1, j):
                left = chart.get((i, k), {})
                right = chart.get((k, j), {})
                for lcat, lsem in left.items():
                    for rcat, rsem in right.items():
                        # Forward application: (X/Y)  Y -> X   [functor LEFT]
                        if (isinstance(lcat, Fun) and lcat.side == "fwd"
                                and lcat.arg == rcat and rsem[0] == "sat"):
                            ncat, nsem = _apply((lcat, lsem), rsem)
                            entries.setdefault(ncat, nsem)
                        # Backward application: Y  (X\Y) -> X   [functor RIGHT]
                        if (isinstance(rcat, Fun) and rcat.side == "bwd"
                                and rcat.arg == lcat and lsem[0] == "sat"):
                            ncat, nsem = _apply((rcat, rsem), lsem)
                            entries.setdefault(ncat, nsem)
            if entries:
                chart[(i, j)] = entries

    full = chart.get((0, n), {})
    lfs = set()
    for cat, sem in full.items():
        if cat == want and sem[0] == "sat":
            lfs.add(frozenset(sem[2]))
    return lfs


def cky_parses(cells):
    """True iff at least one full-span S derivation exists (parse success)."""
    return len(cky(cells, S)) > 0


# --------------------------------------------------------------------------
# AM-algebra compositor (directionless): role-based apply/modify.
#
# AM algebra's apply/modify carry no slash direction. With direction absent the
# type system alone does not determine where a modifier attaches, so the
# compositor resolves it by a NON-positional rule. The faithful canonical choice
# (the failure mode directionality is introduced to fix) is to attach a modifier
# to the clause's head term -- the subject. That single choice is what produces
# the directional asymmetry on position-shift categories (paper Fig 2).
# --------------------------------------------------------------------------
def am_compose(roles):
    """`roles` dict: subj, obj (| None), verb (pred str | None), loc (| None),
    prep (| None). Returns LF frozenset using AM's canonical attachment."""
    edges = set()
    verb = roles.get("verb")
    subj = roles.get("subj")
    obj = roles.get("obj")
    if verb is not None and subj is not None:
        edges.add((verb, subj, obj))
    loc = roles.get("loc")
    prep = roles.get("prep")
    if loc is not None and prep is not None:
        # AM canonical rule: a directionless modifier attaches to the clause
        # head (subject), since surface position is not encoded.
        edges.add((prep, subj, loc))
    return frozenset(edges)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def lf_exact_match(pred_lfs, gold):
    """Exact match iff any predicted LF equals the gold LF set."""
    return gold in pred_lfs if pred_lfs else False


def directional_asymmetry(acc_subj, acc_obj):
    """Fig-2 style asymmetry: |acc on subj-side  -  acc on obj-side|."""
    return abs(acc_subj - acc_obj)
