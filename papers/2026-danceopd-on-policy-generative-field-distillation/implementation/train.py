#!/usr/bin/env python3
"""
DanceOPD: On-Policy Generative Field Distillation — Training Script

Phase 1: Pre-train three frozen teacher models on separate capability buckets.
Phase 2: Distill a single student from all teachers via DanceOPD.
Phase 3: Evaluate student vs. teachers on all distributions.
"""

import time
import math
import torch
import torch.nn.functional as F
from model import SmallFlowModel, flow_matching_loss, euler_rollout, sample_query_state
from data import build_capability_datasets, get_batch_from_bucket

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
DATA_DIM = 2
N_CLASSES = 3
D_MODEL = 64
N_LAYERS = 4

# Teacher pre-training
TEACHER_STEPS = 3000
TEACHER_LR = 3e-3
TEACHER_BATCH = 256
TEACHER_LOG_EVERY = 500

# DanceOPD distillation
DANCEOPD_STEPS = 4000
DANCEOPD_LR = 3e-3
DANCEOPD_BATCH = 128
DANCEOPD_ROLLOUT_STEPS = 16
DANCEOPD_LOG_EVERY = 400
BETA_ALPHA = 5.0
BETA_BETA = 2.0

# Evaluation
EVAL_SAMPLES = 500
EVAL_ROLLOUT_STEPS = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_one_teacher(
    datasets: dict,
    class_id: int,
    steps: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    log_every: int,
) -> SmallFlowModel:
    """Pre-train a single teacher on its capability bucket using flow matching."""
    model = SmallFlowModel(
        data_dim=DATA_DIM,
        n_classes=N_CLASSES,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    name = f"Teacher {class_id}"
    print(f"\n{'='*60}")
    print(f"  Training {name}  ({steps} steps, lr={lr})")
    print(f"{'='*60}")

    for step in range(1, steps + 1):
        model.train()
        x_0 = get_batch_from_bucket(datasets, class_id, batch_size, device)
        noise = torch.randn_like(x_0)
        t = torch.rand(batch_size, device=device)
        cond = torch.full((batch_size,), class_id, dtype=torch.long, device=device)

        loss = flow_matching_loss(model, x_0, noise, t, cond)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % log_every == 0 or step == 1:
            print(f"  [{name}] step {step:>5d}/{steps}  loss = {loss.item():.6f}")

    print(f"  [{name}] done.  Final loss = {loss.item():.6f}")
    return model


@torch.no_grad()
def evaluate_velocity_error(
    model: nn.Module,
    teachers: dict,
    datasets: dict,
    device: torch.device,
    n_samples: int = 256,
):
    """Measure average velocity mismatch between model and each teacher."""
    model.eval()
    results = {}
    for cid in range(N_CLASSES):
        x_0 = get_batch_from_bucket(datasets, cid, n_samples, device)
        noise = torch.randn_like(x_0)
        t = torch.rand(n_samples, device=device)
        cond = torch.full((n_samples,), cid, dtype=torch.long, device=device)

        t_view = t.unsqueeze(-1)
        x_t = (1.0 - t_view) * x_0 + t_view * noise

        student_vel = model(x_t, t, cond)
        teacher_vel = teachers[cid](x_t, t, cond)
        mse = F.mse_loss(student_vel, teacher_vel).item()
        results[cid] = mse
    return results


@torch.no_grad()
def evaluate_sample_quality(
    model: nn.Module,
    datasets: dict,
    device: torch.device,
    cond_id: int,
    n_samples: int = 200,
    n_rollout_steps: int = 32,
):
    """Generate samples from the model for a given class and measure quality
    as the mean distance to the nearest centroid of the target distribution.
    """
    model.eval()
    z_noise = torch.randn(n_samples, DATA_DIM, device=device)
    cond = torch.full((n_samples,), cond_id, dtype=torch.long, device=device)

    trajectory = euler_rollout(model, z_noise, cond, n_steps=n_rollout_steps)
    samples = trajectory[-1]  # final state (t=0)

    # Compute mean distance to dataset centroids
    ds = datasets[cond_id]
    centroid = ds.data.mean(dim=0).to(device)
    dist = (samples - centroid).norm(dim=1).mean().item()

    # Also compute standard deviation spread
    spread = samples.std(dim=0).norm().item()
    return dist, spread, samples


# ---------------------------------------------------------------------------
# DanceOPD Training Loop
# ---------------------------------------------------------------------------

def train_danceopd_student(
    teachers: dict,
    datasets: dict,
    student_init: SmallFlowModel,
    steps: int,
    lr: float,
    batch_size: int,
    rollout_steps: int,
    beta_alpha: float,
    beta_beta: float,
    device: torch.device,
    log_every: int,
) -> SmallFlowModel:
    """Train student via DanceOPD on-policy field distillation.

    The student is initialized from one of the teacher checkpoints
    (analogous to the paper's recommendation: init from strongest capability).
    """
    # We start from Teacher 0 (arbitrary choice for this toy setup;
    # in the paper you'd start from the most relevant capability).
    student = student_init.to(device)
    for p in student.parameters():
        p.requires_grad_(True)

    optim = torch.optim.AdamW(student.parameters(), lr=lr)

    print(f"\n{'='*60}")
    print(f"  DanceOPD Distillation  ({steps} steps, lr={lr})")
    print(f"  Rollout steps: {rollout_steps}, Query: Beta({beta_alpha},{beta_beta})")
    print(f"  Hard routing over {N_CLASSES} teacher fields")
    print(f"{'='*60}")

    losses = []
    for step in range(1, steps + 1):
        student.train()

        # --- A. Hard route: sample one capability bucket uniformly ---
        route_id = torch.randint(0, N_CLASSES, (1,)).item()

        # --- B. Sample data from that bucket ---
        x_0 = get_batch_from_bucket(datasets, route_id, batch_size, device)
        cond = torch.full((batch_size,), route_id, dtype=torch.long, device=device)

        # --- C. On-policy student rollout (stop-gradient) ---
        z_noise = torch.randn(batch_size, DATA_DIM, device=device)
        with torch.no_grad():
            trajectory = euler_rollout(
                student, z_noise, cond, n_steps=rollout_steps
            )

        # --- D. Semantic-side single query (K=1) ---
        z_bar, t_query, idx_query = sample_query_state(
            trajectory, rollout_steps, batch_size, device,
            beta_alpha=beta_alpha, beta_beta=beta_beta,
        )
        # z_bar already detached via the no_grad rollout

        # --- E. Teacher velocity (frozen) ---
        teacher = teachers[route_id]
        teacher.eval()
        with torch.no_grad():
            target_vel = teacher(z_bar, t_query, cond)

        # --- F. Student velocity ---
        student_vel = student(z_bar, t_query, cond)

        # --- G. Plain velocity MSE ---
        loss = F.mse_loss(student_vel, target_vel)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())

        if step % log_every == 0 or step == 1:
            avg_loss = sum(losses[-log_every:]) / len(losses[-log_every:])
            print(
                f"  [DanceOPD] step {step:>5d}/{steps}  "
                f"loss = {loss.item():.6f}  "
                f"avg({log_every}) = {avg_loss:.6f}  "
                f"route = {route_id}"
            )

    print(f"  [DanceOPD] done.  Final loss = {loss.item():.6f}")
    return student


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  DanceOPD: On-Policy Generative Field Distillation")
    print("  Toy implementation with 2D synthetic data")
    print("=" * 60)

    total_start = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Build datasets
    # ------------------------------------------------------------------
    print("\n--- Building capability datasets ---")
    datasets = build_capability_datasets(seed=42)
    for cid, ds in datasets.items():
        print(f"  Bucket {cid}: {len(ds)} samples, dim={ds.data.shape[1]}, "
              f"mean={ds.data.mean(0).tolist()}")

    # ------------------------------------------------------------------
    # Phase 2: Pre-train frozen teachers
    # ------------------------------------------------------------------
    print("\n=== Phase 1: Teacher Pre-Training ===")
    teachers = {}
    for cid in range(N_CLASSES):
        t0 = time.time()
        teachers[cid] = train_one_teacher(
            datasets, cid,
            steps=TEACHER_STEPS,
            lr=TEACHER_LR,
            batch_size=TEACHER_BATCH,
            device=DEVICE,
            log_every=TEACHER_LOG_EVERY,
        )
        elapsed = time.time() - t0
        print(f"  Teacher {cid} trained in {elapsed:.1f}s")

    # Freeze all teachers
    for cid, teacher in teachers.items():
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        print(f"  Teacher {cid} frozen ✓")

    # ------------------------------------------------------------------
    # Phase 3: DanceOPD Distillation
    # ------------------------------------------------------------------
    print("\n=== Phase 2: DanceOPD Distillation ===")
    # Initialize student from Teacher 0 (mirroring paper's "start from
    # strongest relevant capability")
    student_init = SmallFlowModel(
        data_dim=DATA_DIM,
        n_classes=N_CLASSES,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
    )
    # Copy Teacher 0 weights as initialization
    student_init.load_state_dict(teachers[0].state_dict())
    print("  Student initialized from Teacher 0 weights")

    student = train_danceopd_student(
        teachers=teachers,
        datasets=datasets,
        student_init=student_init,
        steps=DANCEOPD_STEPS,
        lr=DANCEOPD_LR,
        batch_size=DANCEOPD_BATCH,
        rollout_steps=DANCEOPD_ROLLOUT_STEPS,
        beta_alpha=BETA_ALPHA,
        beta_beta=BETA_BETA,
        device=DEVICE,
        log_every=DANCEOPD_LOG_EVERY,
    )

    # ------------------------------------------------------------------
    # Phase 4: Evaluation
    # ------------------------------------------------------------------
    print("\n=== Phase 3: Evaluation ===")

    print("\n--- Velocity Error (MSE vs. teacher) ---")
    print(f"  {'Model':<20} {'Bucket 0':>10} {'Bucket 1':>10} {'Bucket 2':>10} {'Average':>10}")
    print(f"  {'-'*60}")

    # Evaluate each teacher on its own bucket and others
    for cid in range(N_CLASSES):
        errs = evaluate_velocity_error(teachers[cid], teachers, datasets, DEVICE)
        avg = sum(errs.values()) / len(errs)
        print(f"  {'Teacher ' + str(cid):<20} {errs[0]:>10.5f} {errs[1]:>10.5f} {errs[2]:>10.5f} {avg:>10.5f}")

    # Evaluate student
    student_errs = evaluate_velocity_error(student, teachers, datasets, DEVICE)
    student_avg = sum(student_errs.values()) / len(student_errs)
    print(f"  {'Student (DanceOPD)':<20} {student_errs[0]:>10.5f} {student_errs[1]:>10.5f} {student_errs[2]:>10.5f} {student_avg:>10.5f}")

    print(f"\n  Student avg velocity error: {student_avg:.5f}")

    print("\n--- Sample Quality (dist to centroid, lower = better) ---")
    print(f"  {'Model':<20} {'Bucket 0':>10} {'Bucket 1':>10} {'Bucket 2':>10}")
    print(f"  {'-'*60}")

    for cid in range(N_CLASSES):
        row = []
        for eval_cid in range(N_CLASSES):
            dist, spread, _ = evaluate_sample_quality(
                teachers[cid], datasets, DEVICE, eval_cid,
                n_samples=EVAL_SAMPLES, n_rollout_steps=EVAL_ROLLOUT_STEPS,
            )
            row.append(dist)
        print(f"  {'Teacher ' + str(cid):<20} {row[0]:>10.4f} {row[1]:>10.4f} {row[2]:>10.4f}")

    # Student on all buckets
    student_row = []
    for eval_cid in range(N_CLASSES):
        dist, spread, samples = evaluate_sample_quality(
            student, datasets, DEVICE, eval_cid,
            n_samples=EVAL_SAMPLES, n_rollout_steps=EVAL_ROLLOUT_STEPS,
        )
        student_row.append(dist)
    print(f"  {'Student (DanceOPD)':<20} {student_row[0]:>10.4f} {student_row[1]:>10.4f} {student_row[2]:>10.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  3 teachers trained ({TEACHER_STEPS} steps each)")
    print(f"  Student distilled via DanceOPD ({DANCEOPD_STEPS} steps)")
    print(f"  Total wall time: {total_time:.1f}s")
    print(f"  Student avg velocity error across all buckets: {student_avg:.5f}")
    print(f"\n  The student should match or outperform any single teacher on")
    print(f"  average across all buckets — demonstrating capability composition.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
