# OrbitQuant — in my own words

**Paper.** Lee et al., *OrbitQuant: Data-Agnostic Quantization for Image and
Video Diffusion Transformers* (arXiv:2607.02461, 2026). Post-training weight+
activation quantization for diffusion transformers (FLUX, Z-Image, Wan,
CogVideoX).

## In my own words

Diffusion transformers are compute-bound even at batch 1, so weight-only
quantization buys nothing — you have to quantize activations too. The blocker is
that DiT activations have **channel outliers that drift across timesteps, prompts
and CFG branches**, so the activation range is a moving target. Prior DiT-PTQ
methods (SVDQuant, PTQ4DiT, AdaTSQ, ViDiT-Q) chase that drift with
**calibration**: re-collect data for every checkpoint / resolution / modality.

OrbitQuant's thesis: **don't estimate the range — rotate it away.** Take a random
orthogonal rotation Π. A unit vector's coordinates after a Haar rotation are
distributed as the fixed marginal `f_d` (one coordinate of a uniform sphere
point) — regardless of the input. So you can precompute, offline, one
MSE-optimal scalar **Lloyd–Max codebook** for `f_d`, and reuse it for every
weight row and every activation token of dimension `d`. Because the codebook is
built on the *post-rotation* marginal, and the rotation is orthogonal, the
rotation **cancels inside the matmul**: `WΠᵀ · Πx = Wx`. The weight absorbs
`Πᵀ` (offline), the activation applies `Π` (one forward rotation online), and
the result is `≈ Wx` with no inverse rotation at runtime.

The structured rotation they actually deploy is **RPBH** (Randomized Permuted
Block-Hadamard): block-diagonal Walsh–Hadamard with random signs, preceded by a
uniform random permutation. Proposition 1 proves that the permutation
**equalizes per-block mass** so every rotated coordinate has variance within
`[1/d(1±ρ)]` of the uniform value — i.e. even outlier-heavy inputs land on `f_d`.
That is the whole reason a *single* calibration-free codebook works for *all*
inputs, and the reason removing the permutation degrades low-bit quality.

## What I learned

- The cleanest single idea: a random rotation turns the data-dependent
  "what range do I quantize over?" problem into the data-independent "quantize
  the sphere-coordinate marginal `f_d`" problem. The codebook is a property of
  the dimension `d`, not of the data. That is a genuinely different axis of
  attack from outlier-suppression / calibration.
- `f_d` is just the symmetric Beta(1/2, (d−1)/2) density on [−1,1] — one
  coordinate of a uniform sphere point — so it has mean 0, variance exactly
  `1/d`, and is already tightly approximated by `N(0,1/d)` for `d≥64`. No
  fitting, no data: it is a Gamma function.
- The rotation-cancels identity is exact and trivial (`ΠᵀΠ = I`) but it is the
  engineering hinge: it makes the online cost a *single forward rotation* rather
  than a quantize→dequantize→rotate-back codec. Verified to 7e-16.
- Lloyd–Max on `f_d` crushes a uniform grid at low bit-width (2664× lower MSE at
  `b=2`), and the advantage shrinks as `b` grows — exactly the regime
  OrbitQuant targets (W2A4). Centroid cell-mean optimality holds to ~0.

## Surprises

- **The paper's Eq-2 normalizer is wrong as rendered.** `sqrt(Γ(d/2)/(πΓ((d−1)/2)))`
  integrates to `sqrt(Γ((d−1)/2)/Γ(d/2))` ≈ 0.21 for d=1024, not 1. The correct
  symmetric-Beta normalizer is `Γ(d/2)/(√π·Γ((d−1)/2))`. I only caught this
  because my explicit "does f_d integrate to 1?" self-check failed; the KS tests
  and the Lloyd–Max codebook (which normalize the lattice weights, hence are
  invariant to a constant rescale of the density) were unaffected. A silent
  constant-factor density typo that the downstream machinery happily ignores.
- **Proposition-1 ρ has an imaginary sqrt in the paper's notation.** It writes
  `sqrt((4k/d) log δ)` with δ∈(0,1) — `log δ < 0`, so the sqrt is of a negative
  number. The intended Hoeffding-without-replacement form is `log(1/δ)`. Easy to
  miss if you only read the bound qualitatively.
- The permutation's low-bit benefit is **small on average and decisive only in
  the worst case.** With outliers placed at random, RPBH vs Block-Hadamard
  (no-perm) land within noise at W2A4. The gap shows up sharply only when
  outliers **co-occur in one block** (the Lemma-2 mass-balancing regime): there
  no-perm is 1.20× worse. This matches the paper's own small Table-3 gap
  (RPBH 0.595 > Block-RHT 0.558 at W2A4) — the permutation is insurance against
  outlier clustering, not a uniform win.

## Harder than expected

- Making the permutation-matters signal **robust** took care: a single matmul
  with a single random token and a single Π realization is noise-dominated, and
  no-perm can win by chance. Averaging over many sign/permutation draws and
  adding an adversarial same-block-outlier config (where Lemma-2 bites) was
  needed to reproduce the qualified "permutation helps at low bit" claim rather
  than overclaim it.
- Tuning the outlier scale so that per-row RTN **collapses** at W2A4 (rel-err
  > 1, "model broken") while OrbitQuant stays bounded — the contrast that
  produces the paper's "only functional method at W2A4" headline — required
  strong-enough outliers without making the signal all-outlier.

## Honest scope

- This verifies the **quantizer mechanism** on synthetic outlier data, not
  end-to-end GenEval/VBench on real FLUX/Z-Image/Wan backbones. The image/video
  quality numbers (Table 1/2) need full diffusion models and are out of scope.
- All paper latency/memory numbers are **fake-quantization overhead** (codes
  dequantized to BF16, matmul in BF16), not realized low-bit speedup — there is
  no codebook-GEMM kernel. Not reproduced.
- The two genuine prose-vs-table defects the breakdown flagged (§C.3
  seed-robustness std bounds violated by FLUX.1-schnell W4A4 = 0.012 and
  FLUX.1-dev W2A4 = 0.014) are not reproducible in this toy; noted, not
  re-derived.

## Code

`implementation/` — `model.py` (`f_d`, Haar/RPBH rotations, Proposition-1 ρ,
Lloyd–Max + uniform codebooks, OrbitQuant weight/activation ops, RTN),
`data.py` (synthetic outlier unit vectors / weight rows), `train.py`
(5 deterministic checks C1–C5). Run:

```
uv run --with numpy --with scipy python implementation/train.py
```

All five checks PASS: C1 marginal matching, C2 Proposition-1 concentration,
C3 rotation-cancels, C4 Lloyd–Max optimality, C5 W2A4 robustness.

## References

- Lee et al. 2026, *OrbitQuant: Data-Agnostic Quantization for Image and Video
  Diffusion Transformers*, arXiv:2607.02461.
- Inherited ingredients: TurboQuant (Haar rotation + Lloyd–Max sphere-codebook),
  QuaRot / Full RHT (randomized Hadamard), Block-RHT (the no-permutation
  ablation).
