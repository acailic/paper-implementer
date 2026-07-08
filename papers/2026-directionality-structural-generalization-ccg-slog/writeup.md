# Directionality in Structural Generalization — Writeup

**Paper:** On the Role of Directionality in Structural Generalization
**Author:** Zichao Wei
**arXiv:** 2607.02307 (July 2026)

---

## In My Own Words

SLOG (Li et al., 2023) is a compositional-generalization benchmark whose 17 test
categories probe whether a semantic parser generalizes to *structural* shifts —
deeper recursion, a modifier in a different position, a moved wh-element. The
prior SOTA, AM-Parser, uses a BERT supertagger but composes meaning in an
**AM algebra** whose `apply`/`modify` operations **do not encode direction** —
the algebra is word-order-independent, designed for AMR-style graph tasks.
Several SLOG categories, though, *explicitly* hinge on direction (where a
modifier sits, where an argument is extracted from). The paper asks: would a
**directed** type system do better exactly there?

The contribution is a representational choice, not a new model. The author swaps
the symbolic backend for **CCG directed types**: a type `S\NP` requires its NP
argument to the *left* (backward application); the slash direction says which
side of the head an argument lives on. They keep the same BERT-base supertagger
and a deliberately tiny compositor — deterministic CKY with forward/backward
application only, plus a single linear decoder (~30K params). The prediction is
sharp: if directionality is the cause, gains should land *precisely* on the
position-shift categories and be *absent* on recursive-depth ones. The paper
reports exactly this pattern — CCG leads on 5/5 position-shift categories and
trails on 6/6 recursive-depth ones.

## What I Learned

The decisive mechanism is that **slash direction localizes attachment to surface
position**. A post-nominal PP modifier typed `NP\NP` (functor expecting an NP on
its left) attaches to the *immediately-left* NP. So "the boy **in the park**
saw the dog" gets the PP on the subject, and "the boy saw the dog **in the
park**" gets it on the object — automatically, from the same type, because
adjacency differs. The AM algebra has no slash; its `modify` is told to attach a
modifier but not *where*, so it must fall back on a non-positional rule (a
default attachment target). That single missing piece is exactly what produces
the **directional asymmetry** the paper highlights in Figure 2: AM-Parser scores
very differently depending on which side the modifier is on, and CCG flattens
that gap.

I also learned why the paper insists directionality is "non-ablatable": it is
not a parameter of CCG but *the definition* of its combinatory operations.
Removing direction deconstructs the formalism — you no longer have CCG, you have
something AM-algebra-shaped. So the paper substitutes a *pattern-alignment*
argument (gains track the direction-relevant dimension) rather than a true
ablation, and is candid that this is correlational.

## Surprises

- The **asymmetry is trivially reproducible in a dozen lines.** With a
  directionless attach-to-subject rule, AM gets the subject-side modifier right
  and the object-side modifier wrong with probability 1 (|Delta| = 1.0); with a
  directed `NP\NP` type the CKY resolves both correctly (|Delta| = 0.0). The
  Figure-2 phenomenon is not subtle once the symbolic backend is the only moving
  part.
- The **app-only restriction matters.** Pure forward/backward application, with
  no composition or type-raising, is enough for the position family (100% parse
  coverage), which is consistent with the paper's "100% CKY coverage" claim.
- The headline (+/-29.9pp) is **fragile**: the +29.9pp position-shift average is
  ~65% carried by a single category (`RC_iobj_extracted`, 0 -> 96.9); without
  it the average collapses to ~+13.1pp. And the "surpasses AM-Parser" gap
  (5.1pp) is smaller than the winner's own seed std (6.4) — non-significant. The
  paper is unusually candid about all of this in a dedicated limitations
  section.

## Harder Than Expected

- **Modeling AM faithfully without cloning AM-Parser.** AM-Parser is a real
  supertagging+composition system, not a trivial heuristic. I modeled it at the
  *type-system* level (directionless apply/modify, modifier resolved by a
  clause-head rule) — the level at which the paper's controlled comparison
  actually lives — rather than as a full system. The risk is over-simplifying
  AM; the benefit is isolating the one variable (direction) the paper controls.
- **Deciding what NOT to reproduce.** The recursive-depth regression
  (-31.9pp, the CP-recur collapse to ~18%) is the paper's most dramatic number,
  but it is driven by the BERT->DeBERTa encoder upgrade (+22.5pp in recursive
  depth), i.e. the *neural supertagger*, not the symbolic backend. There is no
  way to observe it by swapping only the compositor on a controlled grammar. I
  chose to leave it out and say so in the honest scope rather than fake a
  mechanism.

## Code

- `implementation/model.py` — CCG `Atom`/`Fun` types (fwd/bwd slash), app-only
  `cky()` with closure-carrying semantics (nested functors compose for free),
  `am_compose()` role-based directionless compositor, LF exact-match +
  asymmetry metrics.
- `implementation/data.py` — SLOG-like grammar: base / intransitive /
  subj-side-PP / obj-side-PP, with token-index-grounded gold LFs so directed and
  undirected systems compare against one LF. Word-order-swap probe.
- `implementation/train.py` — four checks: base LF exact match, direction
  encodes word order (directed rejects reordered strings; AM order-blind),
  modifier-position asymmetry (Fig 2: CCG |Delta|=0, AM |Delta|=1), and the
  directional-separation sign on the position family (CCG > AM).

All four checks PASS; pure Python stdlib, sub-second on CPU.

## References

- Wei, Z. (2026). On the Role of Directionality in Structural Generalization.
  arXiv:2607.02307.
- Li et al. (2023). SLOG (Structural Generalization benchmark).
- Weißenhorn et al. (2022). AM-Parser / AM algebra.
- Steedman, J. Combinatory Categorial Grammar.
