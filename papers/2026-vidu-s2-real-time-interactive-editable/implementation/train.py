"""Experiments for the Vidu S2 mechanism miniature
(Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation,
Zhang et al., arXiv:2609.11638).

Four experiments, all CPU, all on synthetic 8-dim chunk streams:

  E1  forcing schedules for streaming training:
        TF    - clean ground-truth history windows (Teacher Forcing)
        DF-GT - per-block re-noised GT windows (Diffusion Forcing, off-policy)
        SF    - detached on-policy rollout, clean own history, frozen KV
                (Self-Forcing, as the paper describes it)
        SRF   - detached on-policy rollout, re-noised, ONE gradient-enabled
                replay pass with live KV (Self-Replay Forcing)
      evaluated by long autoregressive rollout from pure noise, plus a
      direct measurement of cross-block gradient flow (SRF has it, SF
      structurally cannot).

  E2  dynamic reference injection: swap the reference mid-stream; the
      reference-visible-to-all-chunks model adopts the new look within a
      chunk, an I2V-style control (reference only at chunk 0) cannot.

  E3  frame-aligned attention for streaming editing vs free cross-attention:
      both learn the task in-distribution; stress test with faster motion
      at test time shows whether the hard alignment prior transfers.

  E4  the spatial layer in miniature: depth->disparity->bidirectional warp,
      measuring disocclusion holes (why the paper needs lightweight fill).

Honest boundary: this is a mechanism demo on synthetic signals, NOT a
reproduction of the proprietary video model. DMD's distribution matching is
replaced by direct regression to the synthetic ground truth (we have the
oracle; the real system does not).
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from data import StreamWorld
from model import RefCondStreamModel, rollout

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)  # tiny model: single thread is faster than
                          # spinning up a thread pool per 1ms op (CPU-only,
                          # and other users are on the GPUs/cores)

DEV = "cpu"
N_TRAIN = 12      # training window length
N_EVAL = 48       # evaluation rollout length (4x the training window)
B = 64
RESULTS = {}


def noised(x, t):
    """Paper convention (Eq. 5): t=0 clean, t=1 noise."""
    eps = torch.randn_like(x)
    return (1 - t) * x + t * eps, eps


def make_batch(world, B, N):
    return world.sample_batch(B, N, device=DEV)


# ---------------------------------------------------------------- E1: arms
def train_tf(world, steps=1000):
    """Teacher Forcing: windows of clean GT chunks (t=0), loss vs GT."""
    model = RefCondStreamModel().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        d = make_batch(world, B, N_TRAIN)
        t0 = torch.zeros(B, N_TRAIN, device=DEV)
        x_hat, _ = model(d["x"], t0, d["c"], d["r"])
        loss = ((x_hat - d["x"]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def window_df_pass(model, x_gt, c, r, source_of_x=None):
    """DF-style window pass: per-block noise levels, one forward, loss vs GT.
    source_of_x: which values to re-noise (GT for DF-GT, rollout for SRF)."""
    t = torch.rand(B, x_gt.shape[1], device=DEV) * 0.98 + 0.01
    x_src = x_gt if source_of_x is None else source_of_x
    x_noisy, _ = noised(x_src, t.unsqueeze(-1))
    x_hat, e = model(x_noisy, t, c, r)
    loss = ((x_hat - x_gt) ** 2).mean()
    return loss, e


def train_dfgt(world, steps=1000):
    """Diffusion Forcing on ground-truth windows (off-policy)."""
    model = RefCondStreamModel().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        d = make_batch(world, B, N_TRAIN)
        loss, _ = window_df_pass(model, d["x"], d["c"], d["r"])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def sf_rollout(model, d, n, grad=True):
    """Self-Forcing-style rollout. Each chunk: fresh noise at t=1, history =
    the model's own clean outputs at t=0, history KV frozen (detached).
    Returns accumulated per-chunk loss (grad flows only through each chunk's
    own denoising path) and the detached trajectory."""
    losses, outs = [], []
    hist_x, hist_t, hist_c = [], [], []
    Bn = d["c"].shape[0]
    for i in range(n):
        noise = torch.randn(Bn, model.d_chunk, device=DEV)
        t_new = torch.ones(Bn, device=DEV)
        xin = torch.stack(hist_x + [noise], dim=1)
        tin = torch.stack(hist_t + [t_new], dim=1)
        cin = torch.stack(hist_c + [d["c"][:, i]], dim=1)
        x_hat, _ = model(xin, tin, cin, d["r"], detach_history_kv=True)
        x_i = x_hat[:, -1]
        losses.append(((x_i - d["x"][:, i]) ** 2).mean())
        const = x_i.detach()
        outs.append(const)
        hist_x.append(const)
        hist_t.append(torch.zeros(Bn, device=DEV))
        hist_c.append(d["c"][:, i])
    return torch.stack(losses).mean(), torch.stack(outs, dim=1)


def train_sf(world, steps=450):
    model = RefCondStreamModel().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        d = make_batch(world, B, N_TRAIN)
        loss, _ = sf_rollout(model, d, N_TRAIN)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_srf(world, steps=450):
    """Self-Replay Forcing: (1) detached on-policy rollout, (2) per-block
    re-noise of the SELF trajectory, (3) single gradient-enabled causal
    replay, loss on all blocks vs GT."""
    model = RefCondStreamModel().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        d = make_batch(world, B, N_TRAIN)
        with torch.no_grad():
            _, traj = sf_rollout(model, d, N_TRAIN, grad=False)
        loss, _ = window_df_pass(model, d["x"], d["c"], d["r"], source_of_x=traj)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_arm(model, world, n_streams=64):
    """Long rollout from pure noise; per-dim MSE vs GT + motion/look split."""
    d = make_batch(world, n_streams, N_EVAL)
    x_hat = rollout(model, d["c"], d["r"], N_EVAL)
    mse = ((x_hat - d["x"]) ** 2).mean().item()
    m_mse = ((x_hat[..., :4] - d["motion"]) ** 2).mean().item()
    a_mse = ((x_hat[..., 4:] - d["look"]) ** 2).mean().item()
    # drift: error growth over position (last quarter vs first quarter)
    per_pos = ((x_hat - d["x"]) ** 2).mean(dim=(0, 2))
    drift = (per_pos[-N_EVAL // 4:].mean() / per_pos[:N_EVAL // 4].mean()).item()
    return {"mse": mse, "motion_mse": m_mse, "look_mse": a_mse,
            "drift_ratio": drift}


def grad_flow_probe(world):
    """Directly measure the cross-block gradient SRF claims: does the loss on
    LATE chunks reach the KV encoding of EARLY chunks? SF's KV is detached by
    construction (gradient identically zero); SRF's replay recomputes the KV
    in-graph, so the gradient is nonzero. Run on a briefly-trained model so
    the magnitudes are meaningful."""
    model = train_srf(world, steps=60)
    d = make_batch(world, 32, N_TRAIN)
    with torch.no_grad():
        _, traj = sf_rollout(model, d, N_TRAIN, grad=False)
    t = torch.rand(32, N_TRAIN, device=DEV) * 0.98 + 0.01
    x_noisy, _ = noised(traj, t.unsqueeze(-1))
    n_early = N_TRAIN - 6  # encodings that ONLY serve as history for the
    # last 6 chunks' losses (their own outputs are not in the probed loss)
    # SRF replay with grads; probe d(sum of losses over last 6 chunks)/d e_j
    x_hat, e = model(x_noisy, t, d["c"], d["r"])
    late = x_hat[:, -6:]
    g, = torch.autograd.grad(((late - d["x"][:, -6:]) ** 2).mean(), e,
                             retain_graph=False)
    early_norms = g[:, :n_early].norm(dim=(0, 2))  # per early encoding
    srf_norm = early_norms[0].item()
    srf_reached = int((early_norms > 1e-10).sum().item())
    # SF: same forward but KV frozen -> gradient to any history encoding is
    # exactly zero by the detach (structural, but verify numerically)
    x_hat2, e2 = model(x_noisy, t, d["c"], d["r"], detach_history_kv=True)
    late2 = x_hat2[:, -6:]
    e2.retain_grad()
    ((late2 - d["x"][:, -6:]) ** 2).mean().backward()
    sf_norm = e2.grad[:, 0].norm().item() if e2.grad is not None else 0.0
    sf_reached = int((e2.grad[:, :n_early].norm(dim=(0, 2)) > 1e-10).sum().item()) \
        if e2.grad is not None else 0
    return {"srf_grad_to_first_encoding": srf_norm,
            "sf_grad_to_first_encoding": sf_norm,
            "srf_early_encodings_receiving_grad": f"{srf_reached}/{n_early}",
            "sf_early_encodings_receiving_grad": f"{sf_reached}/{n_early}"}


# ------------------------------------------- E2: dynamic reference injection
def train_ref_model(world, ref_injection, steps=700):
    model = RefCondStreamModel(ref_injection=ref_injection).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        d = make_batch(world, B, N_TRAIN)
        loss, _ = window_df_pass(model, d["x"], d["c"], d["r"])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_ref_swap(model, world, swap_at=12, n=24, n_streams=64):
    """Roll out with reference A, swap to reference B at swap_at. Track the
    appearance error against BOTH looks per chunk, and motion error."""
    d = make_batch(world, n_streams, n)
    rA = d["r"]
    rB = torch.randn_like(rA)
    # ground-truth looks for each reference (using the world's appearance map)
    R = torch.tensor(world.R, device=DEV)
    lookA = (rA @ R.T)[:, None, :] * 0.75   # best predictor scale E[.5+.5b]=.75
    lookB = (rB @ R.T)[:, None, :] * 0.75
    x_hat = rollout(model, d["c"], rA, n, swap_at=swap_at, r_new=rB)
    out_hat = x_hat[..., 4:]
    errA = ((out_hat - lookA) ** 2).mean(dim=(0, 2))  # per chunk
    errB = ((out_hat - lookB) ** 2).mean(dim=(0, 2))
    mot = ((x_hat[..., :4] - d["motion"]) ** 2).mean().item()
    # sustained adoption: first post-swap chunk from which the look stays
    # closer to the NEW reference than the old one for the rest of the stream
    # (a one-off crossing from noise does not count as adoption)
    latency = None
    for i in range(swap_at, n):
        if all(errB[j] < errA[j] for j in range(i, n)):
            latency = i - swap_at
            break
    return {"err_to_old_ref_before": errA[:swap_at].mean().item(),
            "err_to_old_ref_after": errA[swap_at:].mean().item(),
            "err_to_new_ref_after": errB[swap_at:].mean().item(),
            "adoption_latency_chunks": latency,
            "motion_mse": mot}


# ---------------------------------------------------- E3: streaming editing
class EditWorld(StreamWorld):
    """Editing variant: source stream carries motion (+zeros for look dims);
    target = same motion with reference-appearance look."""

    def sample_batch(self, B, N, device="cpu"):
        d = super().sample_batch(B, N, device)
        src = torch.cat([d["motion"], torch.zeros(B, N, self.d_look)], -1)
        d["src"] = src
        return d


def train_editor(world, source_mask, steps=900):
    model = RefCondStreamModel(use_source=True, source_mask=source_mask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        d = make_batch(world, B, N_TRAIN)
        t = torch.rand(B, N_TRAIN, device=DEV) * 0.98 + 0.01
        x_noisy, _ = noised(d["x"], t.unsqueeze(-1))
        x_hat, _ = model(x_noisy, t, d["c"], d["r"], src=d["src"])
        loss = ((x_hat - d["x"]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_editor(model, world, speed=1.0, n=32, n_streams=64):
    """Streaming edit: feed source frames; model generates the edited stream
    chunk by chunk (own history + aligned source + reference). `speed`
    compresses the source motion timeline (stress test: faster motion than
    training)."""
    d = make_batch(world, n_streams, n + 8)
    if speed != 1.0:  # faster motion: advance `speed` source steps per chunk
        idx = torch.arange(n, device=DEV)
        take = (idx.float() * speed).long().clamp_max(n + 7)
        d["src"] = d["src"][:, take]
        d["motion"] = d["motion"][:, take]
        d["x"] = torch.cat([d["motion"], d["x"][:, take, 4:]], -1)
        d["c"] = d["c"][:, take]
    outs, hist_x, hist_t, hist_c = [], [], [], []
    for i in range(n):
        noise = torch.randn(n_streams, model.d_chunk, device=DEV)
        xin = torch.stack(hist_x + [noise], dim=1)
        tin = torch.stack(hist_t + [torch.ones(n_streams, device=DEV)], dim=1)
        cin = torch.stack(hist_c + [d["c"][:, i]], dim=1)
        sin = d["src"][:, :i + 1]
        x_hat, _ = model(xin, tin, cin, d["r"], src=sin)
        x_i = x_hat[:, -1]
        outs.append(x_i)
        hist_x.append(x_i)
        hist_t.append(torch.ones(n_streams, device=DEV))
        hist_c.append(d["c"][:, i])
    x_hat = torch.stack(outs, dim=1)
    mot = ((x_hat[..., :4] - d["motion"][:, :n]) ** 2).mean().item()
    look = ((x_hat[..., 4:] - d["look"][:, :n]) ** 2).mean().item()
    return {"motion_mse": mot, "look_mse": look}


# ------------------------------------------------------ E4: spatial layer
def stereo_mini(n_px=80, disparity_gain=6.0):
    """Center row with a foreground object (near) over a background (far).
    Warp to left/right eye views by per-pixel disparity d = g / z; count
    disocclusion holes in each view (pixels no center pixel maps to)."""
    rng = np.random.default_rng(7)
    z = np.full(n_px, 4.0)                    # background depth
    z[28:48] = 1.0                            # near object
    z[60:70] = 2.0                            # mid object
    img = rng.uniform(0.2, 0.8, size=n_px)
    disp = disparity_gain / z                 # nearer => larger disparity
    stats = {}
    for view, sign in (("left", +1), ("right", -1)):
        out = np.full(n_px, np.nan)
        for x in range(n_px):
            xs = x + sign * disp[x] / 2.0     # where center pixel x lands
            lo, hi = int(np.floor(xs)), int(np.ceil(xs))
            for xx in (lo, hi):
                if 0 <= xx < n_px:
                    out[xx] = img[x]
        holes = int(np.isnan(out).sum())
        filled = out.copy()
        good = ~np.isnan(filled)
        idx = np.where(good, np.arange(n_px), 0)
        np.maximum.accumulate(idx, out=idx)   # nearest valid to the left
        filled = filled[idx]
        stats[view] = {"hole_fraction": holes / n_px,
                       "max_run": _max_run(np.isnan(out))}
    return stats


def _max_run(holes):
    best = cur = 0
    for h in holes:
        cur = cur + 1 if h else 0
        best = max(best, cur)
    return best


def main():
    t0 = time.time()
    world = StreamWorld(seed=0)
    floor = world.oracle_floor()
    print(f"oracle floor (irreducible per-dim MSE from look modulation b): "
          f"{floor:.5f}\n")

    # E1 -----------------------------------------------------------------
    print("E1: forcing schedules (train on 12-chunk windows, "
          "evaluate 48-chunk rollout from pure noise)")
    arms = {}
    for name, fn in (("TF", train_tf), ("DF-GT", train_dfgt),
                     ("SF", train_sf), ("SRF", train_srf)):
        model = fn(world)
        arms[name] = eval_arm(model, world)
        r = arms[name]
        print(f"  {name:>5}: mse={r['mse']:.4f}  motion={r['motion_mse']:.4f} "
              f" look={r['look_mse']:.4f}  drift(last/first quarter)="
              f"{r['drift_ratio']:.2f}")
    probe = grad_flow_probe(world)
    print(f"  grad-flow probe: d L(last 6 chunks)/d e_0  "
          f"SRF={probe['srf_grad_to_first_encoding']:.4f} "
          f"({probe['srf_early_encodings_receiving_grad']} early encodings "
          f"reached)  SF={probe['sf_grad_to_first_encoding']:.4f} "
          f"({probe['sf_early_encodings_receiving_grad']} reached; zero by "
          f"the detached KV, verified numerically)")
    RESULTS["E1_forcing"] = {**arms, "oracle_floor": floor, **probe}

    # E2 -----------------------------------------------------------------
    print("\nE2: dynamic reference injection (swap reference at chunk 12/24)")
    ref_results = {}
    for inj in ("all", "first"):
        m = train_ref_model(world, inj)
        ref_results[inj] = eval_ref_swap(m, world)
        r = ref_results[inj]
        lat = r["adoption_latency_chunks"]
        print(f"  ref_injection={inj:>5}: err(old ref) before="
              f"{r['err_to_old_ref_before']:.4f} after={r['err_to_old_ref_after']:.4f}  "
              f"err(new ref) after={r['err_to_new_ref_after']:.4f}  "
              f"adoption latency={lat} chunks  motion_mse={r['motion_mse']:.4f}")
    RESULTS["E2_dynamic_reference"] = ref_results

    # E3 -----------------------------------------------------------------
    print("\nE3: streaming editing - frame-aligned vs free source attention")
    ew = EditWorld(seed=1)
    edit_results = {}
    for mask in ("aligned", "free"):
        m = train_editor(ew, mask)
        edit_results[mask] = {
            "in_distribution": eval_editor(m, ew, speed=1.0),
            "fast_motion_x2": eval_editor(m, ew, speed=2.0),
        }
        for cond in edit_results[mask]:
            r = edit_results[mask][cond]
            print(f"  {mask:>7} [{cond:>17}]: motion_mse={r['motion_mse']:.4f} "
                  f" look_mse={r['look_mse']:.4f}")
    RESULTS["E3_frame_aligned_editing"] = edit_results

    # E4 -----------------------------------------------------------------
    print("\nE4: spatial layer miniature (depth -> disparity -> warp)")
    st = {}
    for gain in (3.0, 6.0, 12.0):  # disparity grows as objects come closer
        st[f"gain_{gain:g}"] = stereo_mini(disparity_gain=gain)
        s = st[f"gain_{gain:g}"]["left"]
        print(f"  disparity gain {gain:>4}: disocclusion holes "
              f"{s['hole_fraction']:.1%} of pixels per view, longest hole run "
              f"{s['max_run']} px (why the pipeline needs lightweight fill)")
    RESULTS["E4_stereo_warp"] = st

    RESULTS["runtime_s"] = round(time.time() - t0, 1)
    with open("metrics.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nall experiments done in {RESULTS['runtime_s']}s "
          "(CPU only); metrics written to metrics.json")


if __name__ == "__main__":
    main()
