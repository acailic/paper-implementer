# Directionality in Structural Generalization: directed CCG vs undirected AM-algebra

## What this implements

A **toy** demonstration of the symbolic-backend comparison in Wei (2026,
arXiv:2607.02307), "On the Role of Directionality in Structural Generalization."
The paper keeps the BERT supertagger fixed and swaps the **symbolic backend**
between a directed CCG (deterministic CKY, **forward/backward application only**,
sec 3.1) and AM-Parser's **directionless AM algebra** (apply/modify carry no
slash). This implementation isolates exactly that variable on a tiny
SLOG-like English->LF grammar with two compositors built from the same lexical
supertags:

- **CCG** -- directed types `res/arg` (arg on RIGHT) and `res\arg` (arg on LEFT),
  parsed by **app-only CKY**. Slash direction pins attachment to **surface
  position / adjacency**.
- **AM** -- directionless compositor. With no slash the attachment site is
  under-determined by the type, so a modifier is resolved by a non-positional
  (clause-head = subject) rule -- the failure mode directionality is introduced
  to fix.

Logical form = frozenset of predicate-argument edges `(pred, subj_tok_idx,
obj_tok_idx|None)`; each noun's entity is its surface token index, so directed
and undirected systems compare against one grounded gold LF (no isomorphism).

### Key ideas demonstrated

| Concept | How it appears here |
|---------|---------------------|
| Directed types (CCG) | `Fun(res, arg, side)`; fwd app `(X/Y) Y->X`, bwd app `Y (X\Y)->X` |
| App-only CKY (sec 3.1) | `cky()` uses FA + BA only (no composition / type-raising) |
| Direction encodes word order | Reordered strings get **no** S derivation (C2) |
| Position-driven attachment (Fig 2) | PP `Bwd(NP,NP)` attaches to immediately-left NP -> correct on both subj- and obj-side |
| AM algebra directionlessness | `am_compose()` attach-rule is role-based, position-free |
| Modifier-position asymmetry (Fig 2) | AM `\|Delta\|=1.0` (fails shifted side); CCG `\|Delta\|=0.0` |
| Directional-separation sign (sec 4.2) | position family: CCG > AM by +0.50 exact-match (C4) |

## Files

- `model.py` -- CCG type system (Atom/Fun), app-only CKY with closure-carrying
  semantics, AM role-based compositor, LF exact-match + asymmetry metrics
- `data.py` -- SLOG-like grammar generator (base / intrans / subj_pp / obj_pp)
  with token-index-grounded gold LFs; word-order-swap probe
- `train.py` -- 4 verification checks + PASS/FAIL summary
- `requirements.txt` -- none (Python stdlib only)

## How to run

```bash
python train.py        # ~instant on CPU; pure stdlib, no install step
```

## Expected output

```
[C1] base LF exact match           -> PASS   # CCG & AM both 1.000
[C2] direction encodes word order  -> PASS   # CCG rejects 1.000 of reordered; AM order-blind
[C3] modifier-position asymmetry   -> PASS   # CCG |Delta|=0.000, AM |Delta|=1.000
[C4] directional-separation sign   -> PASS   # position family CCG 1.000 vs AM 0.500 (gap +0.500)
```

## Paper claims verified

- **Direction is encoded in the type.** Directed app-only CKY rejects
  word-reordered strings (no full-S derivation); the directionless AM compositor
  is order-blind (C2). This is the definitional property the paper says is
  non-ablatable (sec 4.2).
- **Position-shift asymmetry (Fig 2).** On the modifier-position family the
  directed system is symmetric (PP attaches by surface position -> high on BOTH
  subj- and obj-side, `\|Delta\| ~ 0`); the undirected AM system is asymmetric
  (its non-positional attach rule fails the shifted side -> `\|Delta\| ~ 1`).
  This reproduces the paper's Fig-2 finding that AM-Parser shows large
  directional asymmetry on the modifier pairs and CCG reduces it.
- **Directional-separation sign (sec 4.2).** Across the position family,
  directed exact-match exceeds undirected by a wide margin -- the positive
  (+position-shift) side of the paper's +/-29.9pp pattern, isolated to the
  symbolic backend.

## Honest scope (what is NOT claimed here)

Three headline numbers from the paper are **not** reproduced, by design, because
they are driven by components this toy deliberately does not include:

- **The -31.9pp recursive-depth regression (CP-recur collapse, breakdown H3).**
  In the paper the recursive-depth dimension is fixed by the
  BERT->DeBERTa encoder upgrade (+22.5pp, breakdown H4), i.e. the *neural
  supertagger*, not the symbolic backend. Isolating only the compositor there is
  no recursion collapse to observe; the app-only directed CKY parses the
  controlled position grammar with 100% coverage. Reproducing CP-recur would
  require the BERT supertagger + seed variance, which is the encoder confound.
- **The "surpasses AM-Parser" headline (H1).** 75.9 +/- 6.4 vs 70.8 +/- 4.3 is a
  non-significant seed-std overlap (gap 5.1 < std 6.4); the toy has no seed
  variance so the comparison is not applicable.
- **The +29.9pp magnitude.** The paper's +29.9pp is itself single-category
  (RC_iobj_extracted) carried (H2); the toy measures only the *sign* and the
  asymmetry, on a cleaner PP-attachment family, not SLOG's 17 categories.

## Hardware

CPU, pure Python, sub-second runtime. No GPU, no dependencies.
