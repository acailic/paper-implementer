# Write-up — StateM re-implementation (paper #12)

**Paper:** "StateM: Reaching 95.3% Raw Accuracy, or a $15 Frontier Run, on
Terminal-Bench 2.1 via Harness Scaling" — Qin et al., arXiv:2608.15089.
**What was built:** the StateM control surface, from scratch, stdlib-only
(`implementation/`), demoed end-to-end by a deterministic scripted agent.

## What the re-implementation demonstrates

1. **The goto protocol is the product.** Six ordered steps — edge, checks,
   persistence, guards, commit+log, entry — are enough to make "success
   recorded only at a terminal state" a runtime property rather than a hope.
   The demo shows a failing command check at step 2 keeping the run in the
   source state with the reason visible, exactly as specified.
2. **Checks block; guards route.** Re-implementing this surfaced the design
   lesson directly: my first runbook put `tests_pass` in BOTH a state check
   and an edge guard — the check fired first and the repair routing never
   ran. The paper's separation is load-bearing: `before_transfer` decides
   *whether you may leave*, the guard decides *where you go next*.
3. **Evidence classes travel with every result.** Command/predicate checks
   carry "host-reproducible"; checklists carry "self-attestation"; llm_review
   carries "semantic-judgment". The runtime cannot accidentally promote a
   structured assertion into proof — the ladder is enforced in the data
   model, not in prose.
4. **Resume is a property of the journal, not of intelligence.** Reloading
   the run yields the authoritative phase from the last committed transition;
   nothing reconstructs state from a trace. This is the mechanism behind the
   paper's 22-hour endurance claim, reproduced in miniature.

## Honest limits

- The benchmark claims (92.1% / 95.3% / 88.1%, $15 final-score evidence,
  transfer hierarchy) are not reproduced: they need Terminal-Bench 2.1,
  frontier APIs, and the authors' evolved runbook. What IS reproduced is the
  substrate those numbers ride on.
- Hook execution is simplified (receipt maps rather than arbitrary
  lifecycle code), and `manual` checks auto-approve in demo mode.
- The authors' harness learning loop (postmortems → executable preconditions)
  is represented by the runbook's static checks, not by an evolving profile.

Status: coding → **done** (12 papers of the top-cited queue complete).
