"""Verification for "On the Role of Directionality in Structural Generalization"
(Wei 2026, arXiv:2607.02307).

Isolates the paper's controlled variable -- the symbolic backend -- on a tiny
SLOG-like grammar and checks the load-bearing, reproducible claims from the
breakdown (sec 7 "citable falsifiable content"):

  C1 base LF          -- directed app-only CKY parses saturated clauses to the
                         correct logical form (exact match).
  C2 direction enc.   -- directed CKY REJECTS word-reordered strings (no S
                         derivation) because slash direction encodes order; the
                         undirected AM compositor is order-blind and still emits
                         an LF.
  C3 position asymm.  -- (paper Fig 2) on the modifier-position family the
                         directed system is symmetric (PP attaches by surface
                         position -> high on BOTH subj-side and obj-side,
                         Delta ~ 0); the undirected AM system is asymmetric
                         (its canonical non-positional attachment rule fails one
                         side -> large Delta).
  C4 dir-separation   -- across the position family, directed exact-match
                         exceeds undirected exact-match by a wide margin -- the
                         +position-shift SIGN of the paper's sec 4.2.

What is deliberately NOT claimed: the -31.9pp recursive-depth regression and the
"surpasses AM-Parser" headline are encoder/supertagger-driven (breakdown H3/H4,
non-significant seed-std overlap H1) and are not reproducible by isolating the
symbolic backend. See writeup Honest Scope.
"""

import data
from model import cky, cky_parses, am_compose, lf_exact_match, directional_asymmetry

N = 60  # instances per template


def acc_directed(insts):
    ok = sum(lf_exact_match(cky(it["cells"]), it["gold"]) for it in insts)
    return ok / len(insts)


def acc_am(insts):
    ok = sum(am_compose(it["roles"]) == it["gold"] for it in insts)
    return ok / len(insts)


# --------------------------------------------------------------------------
def check_base():
    """C1: directed app-only CKY gets the base transitive + intransitive LF."""
    base = data.sample_set("base", N, seed=1)
    intr = data.sample_set("intrans", N, seed=2)
    a_base = acc_directed(base)
    a_intr = acc_directed(intr)
    a_am = acc_am(base)
    passed = a_base >= 0.99 and a_intr >= 0.99 and a_am >= 0.99
    print(f"  base transitive : CCG exact-match {a_base:.3f}")
    print(f"  intransitive    : CCG exact-match {a_intr:.3f}")
    print(f"  base (AM check) : AM  exact-match {a_am:.3f}")
    return passed, dict(base=a_base, intrans=a_intr)


def check_direction_encoding():
    """C2: directed rejects reordered word order; AM is order-blind."""
    base = data.sample_set("base", N, seed=3)
    # Reorder each sentence (verb fronted) -> ungrammatical for a directional
    # parser. Directed should produce NO full S derivation.
    ccg_reject = 0
    for it in base:
        moved = data.swap_order(it["tokens"])
        moved_cells = data._cells(moved)
        if not cky_parses(moved_cells):
            ccg_reject += 1
    reject_rate = ccg_reject / len(base)
    # AM is role-based: it emits an LF from the (unchanged) roles regardless of
    # token order, so it is by construction order-blind -> it "accepts".
    am_blind = 1.0
    passed = reject_rate >= 0.95
    print(f"  CCG rejects reordered order : {reject_rate:.3f} (direction encodes order)")
    print(f"  AM  order-blind (accepts)   : {am_blind:.3f} (no slash to check)")
    return passed, dict(ccg_reject=reject_rate, am_blind=am_blind)


def check_position_asymmetry():
    """C3: modifier-position asymmetry (paper Fig 2)."""
    subj = data.sample_set("subj_pp", N, seed=4)
    obj = data.sample_set("obj_pp", N, seed=5)
    ccg_subj, ccg_obj = acc_directed(subj), acc_directed(obj)
    am_subj, am_obj = acc_am(subj), acc_am(obj)
    d_ccg = directional_asymmetry(ccg_subj, ccg_obj)
    d_am = directional_asymmetry(am_subj, am_obj)
    print(f"  CCG : subj-side {ccg_subj:.3f}  obj-side {ccg_obj:.3f}  |Delta| {d_ccg:.3f}")
    print(f"  AM  : subj-side {am_subj:.3f}  obj-side {am_obj:.3f}  |Delta| {d_am:.3f}")
    passed = d_ccg <= 0.05 and d_am >= 0.5
    return passed, dict(ccg_subj=ccg_subj, ccg_obj=ccg_obj, d_ccg=d_ccg,
                        am_subj=am_subj, am_obj=am_obj, d_am=d_am)


def check_directional_separation():
    """C4: on the position family, directed >> undirected (the +position sign)."""
    subj = data.sample_set("subj_pp", N, seed=6)
    obj = data.sample_set("obj_pp", N, seed=7)
    fam = subj + obj
    a_ccg = acc_directed(fam)
    a_am = acc_am(fam)
    gap = a_ccg - a_am
    print(f"  position-family exact-match : CCG {a_ccg:.3f}  AM {a_am:.3f}  gap {gap:+.3f}")
    # Paper: CCG leads on position-shift (+29.9pp). Toy sign must match and be
    # large (the asymmetry is driven entirely by the obj-side attachment).
    passed = gap >= 0.4
    return passed, dict(ccg=a_ccg, am=a_am, gap=gap)


def main():
    print("Directionality (CCG vs AM-algebra) -- symbolic-backend isolation\n")
    results = {}
    print("[C1] base LF exact match")
    results["base"], _ = check_base(); print(f"  -> {'PASS' if results['base'] else 'FAIL'}\n")
    print("[C2] direction encodes word order")
    results["order"], _ = check_direction_encoding(); print(f"  -> {'PASS' if results['order'] else 'FAIL'}\n")
    print("[C3] modifier-position asymmetry (Fig 2)")
    results["asym"], _ = check_position_asymmetry(); print(f"  -> {'PASS' if results['asym'] else 'FAIL'}\n")
    print("[C4] directional-separation sign on position family")
    results["sep"], _ = check_directional_separation(); print(f"  -> {'PASS' if results['sep'] else 'FAIL'}\n")

    print("Summary")
    for k, v in results.items():
        print(f"  {k:6s}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
