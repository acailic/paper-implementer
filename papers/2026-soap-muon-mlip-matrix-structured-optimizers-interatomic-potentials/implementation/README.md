# SOAP and Muon for MLIPs — matrix-structured optimisers vs AdamW

## What this implements

A **toy** demonstration of "Beyond Adam: SOAP and Muon for Faster, Label-Efficient
Training of Machine Learning Interatomic Potentials" (Harari et al. 2026,
arXiv:2607.02499). The paper is an *empirical optimiser bake-off* on NequIP/Allegro
interatomic potentials; this implementation isolates the four optimisers and the
load-bearing mechanisms on small, fully-verifiable synthetic problems:

- **AdamW** (reference baseline, diagonal preconditioning)
- **Muon** (Alg 1): Nesterov momentum + **Newton-Schulz5** orthogonalisation of the update
- **SOAP** (Alg 2): **Shampoo eigenspace** projection + AdamW *inside* the eigenbasis
- **SOAP-Muon** (Alg 2 with `ortho=True, normalize=True`): SOAP + singular-value-power orthogonalisation + RMS norm

Routing follows the paper's Table 2 parameter-group assignment: **2-D (matrix)
parameters get the matrix-structured method; 1-D / scalar parameters get AdamW.**

### Key paper ideas demonstrated

| Concept | How it appears here |
|---------|---------------------|
| Newton-Schulz5 = polar factor | `newton_schulz5` cubic Newton iteration `0.5(3X − X Xᵀ X)`; verified == `U Vᵀ` from SVD to machine precision (check C1) |
| Shampoo eigenspace preconditioning | `SOAP` projects the gradient into the eigenvectors of `L=GGᵀ`, `R=GᵀG`, runs elementwise Adam there, back-projects (check C2) |
| Curvature eigenbasis recovery | the Shampoo `L` statistic recovers the eigenvectors of the true left curvature `AᵀA` (off-diag fraction ≈ 0.02) — *why* SOAP preconditions rotated anisotropy (check C2) |
| Muon orthogonalisation is magnitude-blind | NS collapses every singular value to 1 (spread 1930 → 1 at the limit); consequence: Muon plateaus above SOAP on a rank-1-signal problem (check C3) |
| Label-efficient force supervision | energy+force regression (Eq 1–3) with a tunable force-label fraction; SOAP @ 50% forces matches/beats AdamW @ 100% forces (check C4) |

## Files

- `model.py` — Newton-Schulz5, SVD-power ortho, RMS-norm, and the four optimisers (`AdamW`, `Muon`, `SOAP`, `SOAP-Muon`) over a shared `param.grad` interface
- `data.py` — three synthetic problems: anisotropic Kronecker-quadratic, rank-1-gradient problem, energy+force MLP with sparse force supervision
- `train.py` — four verification checks + summary
- `requirements.txt` — torch, numpy

## How to run

```bash
pip install -r requirements.txt
python train.py
```

(or without installing into the system env: `uv run --with numpy --with torch python train.py`)

The script runs ~30 s on CPU and prints one PASS/FAIL line per check.

## Expected output

```
[PASS] C1 NS5 == polar(SVD): 8x8: ||OᵀO-I||_max=4.4e-16, rel-to-UVᵀ=0.000; ...
[PASS] C2 SOAP recovers curvature eigenbasis: left off-diag=0.020  right off-diag=0.012; horse-race AdamW=7.3e-07 SOAP=1.0e-07
[PASS] C3 Muon magnitude-blindness: mech SV-spread 1930->NS5(5)=257->NS5(limit)=1.00; rank-1 final loss Muon=9.9e-09 > SOAP=5.5e-15 (AdamW=2.6e-10)
[PASS] C4 SOAP@50% force ~= AdamW@100%: force MAE  AdamW@100%=0.1288  SOAP@50%=0.1218  SOAP@100%=0.1310  (SOAP@50%/AdamW@100%=0.946)

4/4 checks PASS
```

## Paper claims verified

- **NS5 orthogonalises** (C1): the Newton-Schulz iteration computes the polar factor `U Vᵀ` to machine precision — the mechanic both Muon and SOAP-Muon (`ρ=0`) rely on.
- **SOAP's preconditioner is curvature-aligned** (C2): the Shampoo row/column covariance recovers the true curvature eigenbasis of a Kronecker-Hessian quadratic (`Qᵀ(AᵀA)Q` is diagonal to within 2% off-diagonal mass). This is the structural reason SOAP preconditions *rotated* anisotropy that diagonal Adam cannot see.
- **Muon can degrade** (C3): orthogonalisation is magnitude-blind (every singular value → 1), so on a rank-1-signal problem Muon's fixed-norm update plateaus an order of magnitude above SOAP/AdamW — the in-vitro signature of the paper's honest-scope finding that "the orthogonalisation step is the primary source of degradation" (Muon is worse than AdamW on water energy).
- **Label efficiency** (C4): with **half** the force labels, SOAP reaches a force MAE **0.95×** AdamW-at-full-force — the paper's CDP headline ("SOAP-Muon @ 50% forces ≈ AdamW @ 100%").

## Honest scope

- These are synthetic proxies, not NequIP/Allegro on water/CDP. The label-efficiency check (C4) uses a coordinate→energy MLP with analytic forces, not a DFT-grade PES.
- The **horse-race** in C2 is reported but **not gated**: on a small quadratic SOAP's Adam-in-eigenspace ≈ diagonal AdamW (both isotropise via the second moment); SOAP's *win* there is modest (~7× lower final loss at the chosen seed/LR) and seed-dependent, matching the paper's observation that the headline gaps come from the trained representation at network scale, not from the score rule on a toy surface.
- The Muon `ns_steps=5` default (paper's cheap budget) gives an *approximate* orthogonalisation; the polar factor is only reached at machine precision with more iterations (C1 uses 20, C3 shows 5 → 60 → limit). SOAP-Muon's `ρ=0.5` SVD-power variant is implemented but the label-efficiency check uses plain SOAP, since SOAP is the paper's recommended drop-in default.
