# EVA-Client — Source-First Breakdown

**Paper:** "EVA-Client: A Unified Framework for Deployment, Evaluation, and Data Collection on Real Robots" (Yang, Yi, Wang, Zhong, Yang, Wu, Bai, Chen, Zhang, Huang, Liu — CoLab, Beihang University).
**arXiv:** 2607.02646v1 [cs.RO], 2 Jul 2026. PDF: 15 pp, 10.7 MB.
**Code:** https://github.com/Noietch/EVA-CLIENT — **Project:** https://colalab.net/projects/eva-client
**Subarea:** real-robot **deployment / data-collection / evaluation infrastructure** for trained manipulation policies (VLAs/VAMs/WAMs). NOT a policy, benchmark, or dataset — a client layer. Genuinely fresh category for this repo: a *systems/infra* paper, not a methods paper. Self-described as "ongoing development" report with "regular updates."

> Sourcing note: every table/equation below is transcribed verbatim from `paper_layout.txt` (`pdftotext -layout`); line ranges cited inline. There are **no numeric benchmark tables** in this paper — all task outcomes (table-tennis rally, cloth fold) are explicit "illustrative observations from real deployments rather than a controlled study" (§9, line 704). Figure-derived filmstrip claims are flagged, not back-filled.

---

## TL;DR

Training-side robot-policy frameworks (openpi, LeRobot, StarVLA, VLA Foundry) have consolidated; the **real-robot deployment side** is still "a scatter of per-policy, per-robot scripts." EVA-Client is a single open-source client that closes that gap — data collection, deployment, chunk smoothing, evaluation, and recorded feedback in one codebase.

- **Three contributions (§1, lines 155–169):** (1) one client covering the whole real-robot iteration loop; (2) a component-decoupled architecture that composes out-of-the-box (signal-source × transport × robot × strategy = an orthogonal grid); (3) reproducible evaluation that doubles as data collection (every run records training-ready rollout data + per-run logs + a comparison viewer).
- **Five narrow-interface layers (§3, lines 285–294):** transport / robot-description / policy-client / inference-strategy / CLI-web. Each replaceable independently via shared dataclass interfaces (observation object, action arrays/chunks, robot-description object).
- **Five execution modes (Table 1, lines 199–216):** open-loop sim → real single-chunk stepping → segmented sim-to-real → continuous execution → data collection.
- **Five inference strategies (Table 2, lines 538–548):** synchronous, async prefetch (linear-overlap blend), temporal ensemble (ACT-style exp-weighted), naive async (chunk replace, ablation baseline), Real-Time Chunking (RTC = overlap + server-side conditioning).
- **Two formal equations (§6):** Eq 1 linear-overlap blend (line 523); Eq 2 ACT-style exponentially weighted temporal ensemble (lines 582–584).
- **Continuous IK (§4, lines 364–385):** PyRoki (JAX) Levenberg–Marquardt, trust-region + dense-Cholesky, warm-started frame-by-frame, three weighted costs (pose / rest / velocity) under hard joint-limit constraint.
- **Three logged action streams (§8, lines 685–691):** raw per-chunk predictions / smoothed post-strategy actions / executed commands — tagged per source chunk, so a bad motion is attributable to model vs smoothing vs execution.

---

## 1. The gap EVA-Client closes (§1, lines 117–178)

Deployment is "still largely underappreciated." Taking a checkpoint from "it trained" to "it moves the robot correctly" needs a deployment stack beyond querying a policy server: observation/command interfaces, real-time action scheduling, latency compensation, action-space conversion, rollout logging. These are "not usually treated as model contributions" but "often determine whether a trained policy can run safely, smoothly, and reproducibly." Today they are "method- or robot-specific scripts" (e.g. per-model clients bundled with openpi / LeRobot).

**Three consequences of the gap (lines 125–136):**
1. **Robot-integration** lies below the abstraction level of model/training frameworks — control loop depends on robot-specific cameras, state feedback, middleware, action spaces; code rarely transfers across robots.
2. **Real-time-execution** is coupled to deployment not training — chunking, async inference, latency compensation, smoothing are method-specific implementation details, hard to inspect/compare on hardware.
3. **Physical-evaluation** is shaped by the deployment pipeline — hard to tell whether an improvement is a better model or a different deployment setup; demonstrations a run could yield are "usually lost rather than routed back to the next round of external training."

**Explicit scope statement (lines 174–178):** EVA-Client is a *client and deployment layer* for trained policies. Assumes a policy server already exists; **model-agnostic** w.r.t. that server. Not itself a policy/benchmark/dataset. IK targets **serial-arm manipulators** only. Positioned as the "deployment counterpart to the now-mature training frameworks."

---

## 2. Five-layer architecture (§3, lines 245–308)

High-level (Figure 2, lines 262–279): EVA-Client is a **thin client between signal sources and robot execution**.

- **Signal sources (left):** trained policy servers — openpi, StarVLA, GR00T, Dream-Zero — plus human teleoperation. Model-based sources receive **instructions + observations**, return **actions or action chunks**; teleop supplies actions directly.
- **EVA-Client (center):** exposes Debug / Collect / Evaluation workflows.
- **Robot execution (right):** turns actions into robot commands over **ROS 1/2 or ZMQ**, returns synchronized observations.

**Three design principles (lines 246–253):** (1) Robot-agnostic — control loop, inference strategies, debug modes assume no particular robot/middleware/action-representation; (2) Reproducible — a deployment is fully described by a configuration, every run records what was observed/inferred/executed; (3) Observable — inspect/intervene at each stage, preview in sim, step one chunk, visualize trajectories, without rewriting code.

**The five layers (lines 285–294):** every workflow drives the same layers, differing only in CLI/web presentation.

| Layer | Responsibility |
|---|---|
| **Transport** | abstracts how synchronized observations are captured and commands delivered — real robot / dataset / socket fake node |
| **Robot description** | declares actuator groups, cameras, observation schema, topic mappings, optional kinematics; registers in a registry |
| **Policy client** | queries a model-based source, returns an action chunk; teleop sources feed the collection path |
| **Inference strategy** | decides when to call the policy and how to turn overlapping chunks into one smooth action stream (§6) |
| **CLI/web** | drives the session state machine; provides interactive/collection/evaluation front-ends |

Layers communicate through small dataclass interfaces — **shared observation interface, action arrays/chunks, robot-description object** — so each is independently replaceable.

**Control loop (lines 296–303):** closed cycle. Each iteration: read synchronized observation → (model source) forward observation + language instruction to policy server, receive action chunk, issue commands; (teleop) source supplies operator actions through same execution+logging path. Robot produces next observation, closes loop. Runs at a **fixed control rate deliberately decoupled from policy-query rate**. Chunk scheduling/blending = inference strategy (§6). One **session state machine** (idle → ready → running) governs setup/start/stop/reset uniformly across workflows.

**Configuration (lines 305–308):** a deployment is fully specified declaratively — one config file fixes robot, transport backend, policy endpoint, action-space modes, loop rates, prompts, inference strategy. CLI flags override individual fields. A run is "reproducible from its configuration alone" — this is what §8 records to reconstruct later.

---

## 3. Backends and hardware (§4, lines 311–385)

### 3.1 Transport backends (lines 324–339)

| Backend | Behavior |
|---|---|
| **ROS1** | subscribes to camera + joint-state topics; time-synchronizes into observation frames. Does NOT use the ROS message-sync utility with tolerance window — instead **buffers each topic in a bounded deque** (length = configurable staleness window), aligns to latest timestamp common to all required streams, drops older messages. If no common timestamp in current buffers, **waits for re-sync** rather than emit a mismatched frame. Publishes joint commands to real robot (+ optional parallel sim topics). |
| **ROS2** | mirrors ROS1 design on the ROS2 client library: per-topic buffering, header-stamp alignment, **best-effort QoS** for live streams. ROS2-native robots handled directly, not bridged. |
| **Dataset** | replays a **LeRobot-format** episode (parquet observations + MP4 video) — exercises an inference pipeline **open-loop with no hardware**. |
| **ZMQ** | lightweight wire protocol over a socket; pointed at a bundled **fake execution-layer node**, synthesizes observation frames in a separate process — **development without a physical robot**. |

> **ROS-decoupling note (lines 337–339):** removing the policy-server dependency is handled separately (bundled fake policy server, or server-free mock/replay policy clients) — NOT by the transport layer. **ROS is required only for the matching ROS backend; ZMQ and dataset backends have no ROS dependency.**

### 3.2 Robot support (lines 341–348)

A robot is **declared, not coded**: a robot-description object lists actuator groups (arms, grippers, base), camera/observation schema, topic mappings, optional kinematics, and registers itself. **Reference implementation = dual-arm Piper** (two 6-DoF arms + grippers) → **14-D action vector = 2 × (6 joints + 1 gripper)**. Same generic backend already connects to **6 platforms**: Franka, UR5e, Galaxea R1-lite, AgileX Piper, AgiBot G2, ARX R5. IK via shared robot-agnostic **PyRoki** solver. Supporting a new robot = writing one description class.

### 3.3 Camera support (lines 350–352)

ROS-based robots: cameras are ordinary topics consumed by the transport layer. Non-ROS robots need an alternative image source — **middleware-independent camera interface for non-ROS robots is a current limitation** (§9).

### 3.4 Action spaces + inverse kinematics (lines 354–362)

EVA-Client **decouples three action spaces usually conflated:**
1. **observation state space**
2. **policy output space**
3. **publication space**

Each may independently be **joint or end-effector**. In end-effector mode each arm = position + orientation + gripper command; orientation accepted as quaternion / roll-pitch-yaw / 6D rotation, canonicalized internally. Scalar gripper dimension optionally **snapped to open/closed at a configurable threshold** for binary grippers; continuous grippers pass through unchanged.

When policy outputs end-effector poses but robot is commanded in joint space → **IK solver converts poses to joint targets on the fly**. Lets you feed joint-space observations to a pose-predicting policy and still command a joint-controlled arm, without changing policy or robot driver.

### 3.5 Continuous IK (lines 364–385) — the most formal §4 content

Built on **PyRoki** (Kim et al., 2025), a **JAX-based kinematics toolkit**, using its nonlinear least-squares solver. Both inverse and forward kinematics come from PyRoki's differentiable forward-kinematics routine; **one backend serves every robot in the zoo.**

- **Per-frame solve:** each frame of an action chunk = a single-configuration least-squares problem solved by **Levenberg–Marquardt**, using a **trust-region step with dense-Cholesky factorization**.
- **Three weighted costs under a hard joint-limit constraint:**
  - **end-effector pose cost** — analytic Jacobian, separate position/orientation weights;
  - **rest cost** — biases toward the home pose;
  - **velocity cost** — penalizes only over-speed steps.
- **Continuity:** frames solved sequentially; **each frame warm-started from the previous solution**; **first frame anchored to measured joint state** so execution begins without a jump. Velocity cost biases each solve toward the nearby IK branch, keeping targets on a mostly-continuous joint trajectory. Near singularities or joint limits, discontinuities can still occur and **are logged**.
- **Configurable solver params:** separate position/orientation, rest, velocity cost weights + tracking-error tolerance. A frame whose tracking error exceeds tolerance is **still commanded but flagged for review**, not rejected.
- Complements §6 inference-strategy smoothing: continuous IK keeps **joint targets** on a coherent trajectory; inference strategies blend **overlapping chunks** into a smooth command stream.

---

## 4. Operation model + execution modes (§5, lines 388–414)

Deployment treated as **a debugging activity, not a one-shot launch**. Whole operation model exposed through a single **web console** (Figure 3, lines 447–459): persistent tab bar → 5 tabs (**Debug, Collect, Eval, Replay, read-only Result**). All share one layout: control panel + Viser-based (Yi et al., 2025) 3D scene + time-synchronized camera streams. Viser 3D scene renders joint state, end-effector poses, executed trajectories live; doubles as render target for open-loop and sim-to-real modes.

### Table 1 — Debugging and execution modes (verbatim, lines 199–216)

| Mode | Drives | Granularity | Primary use |
|---|---|---|---|
| Open-loop simulation | Sim | Continuous | Inspect policy behavior with no hardware risk |
| Real single-chunk stepping | Real robot | One chunk | Step through execution on hardware, chunk by chunk |
| Segmented sim-to-real | Sim then real | One chunk | Preview a chunk in sim, then confirm it on the real robot |
| Continuous execution | Real robot | Continuous | Real-time deployment at full control rate |
| Data collection | Real robot | Continuous | Record teleoperated demonstrations into training-ready datasets |

> Evaluation (§8) runs on top of these modes; data collection (§7) reuses the same console and backends.

- **Open-loop simulation (lines 399–401):** policy runs against live/synthetic observations but actions routed **exclusively to Viser 3D visualization**, not hardware. Safest entry point — confirm a checkpoint produces sane motion before physical risk.
- **Real single-chunk stepping (lines 403–405):** on physical robot, advance exactly one action chunk per command; between chunks robot is stationary, user inspects — localizes failures (e.g. bad grasp pose) to a specific inference step.
- **Single-chunk sim-to-real stepping (lines 407–410):** interleaves the two — each chunk first previewed in sim, then user commits to real robot or cancels. Human checkpoint between prediction and physical execution.
- **Continuous execution (lines 412–414):** full real-time deployment — control loop publishes actions continuously at configured rate, selected inference strategy (§6) handles chunk scheduling/smoothing in background. Mode used for actual task execution.

---

## 5. Inference strategies (§6, lines 417–619) — the paper's formal core

Real-time execution of chunk-based policies: each inference call returns a **horizon of future actions** `A^(j) = (a_0^(j), a_1^(j), …, a_{H−1}^(j))` (lines 424–428), where `H` = prediction horizon and each `a^(j) ∈ ℝ^d`. Consecutive chunks overlap in time and may disagree at boundaries → strategy decides **when to request new chunks, which delayed actions to trust, how to combine overlapping predictions.**

> **Why it matters (Figure 4, lines 429–503):** same EVA-Client deployment drives two contrasting tasks — high-dynamics **table tennis** (Figure 4a) and long-horizon **cloth folding** (Figure 4b) — changing only inference strategy + robot description, no task-specific code. ⚠ *Figure-derived / illustrative, not a controlled benchmark:* under synchronous execution the pause-and-go stall "keeps the arm from tracking the fast ball, so the rally does not get going in our deployment"; async scheduling "lets the rally proceed." Folding "runs under asynchronous execution." Authors stress these are "illustrative observations from real deployments rather than a controlled benchmark" (line 465).

### Table 2 — Inference strategies (verbatim, lines 538–548)

| Strategy | Async | Smoothing |
|---|---|---|
| Synchronous | no | − |
| Async prefetch | yes | Linear Overlap |
| Temporal Ensemble | yes | Exponentially Weighted |
| Naive Asynchronous | yes | Chunk Replace |
| Real-Time Chunking (RTC) | yes | Overlap + Server |

Smoothing shorthands (lines 539–541): linear overlap = latency-trimmed chunks blended over overlap window; exp. weighted = ACT-style exponentially weighted temporal ensembling; chunk replace = newest chunk replaces buffer with no blending; overlap+server = RTC server-side conditioning + final linear-overlap pass.

### 5.1 Synchronous execution (lines 468–471)

Request chunk → optionally crop to execution horizon → execute all actions in order → request next chunk. Simple, exactly reproducible, but **robot pauses during every forward pass** → pause-and-go motion. Useful for debugging and for policies fast enough that the pause is negligible.

### 5.2 Async prefetch with linear-overlap blending (lines 473–528)

Inference + control run in parallel; implements the **latency-aware overlap-blending scheme of Yu et al. (2026)**. Background thread keeps requesting chunks; main control loop executes actions from a buffer at the robot control rate. Avoids blocking the robot during a forward pass, but introduces a **temporal alignment problem**: when a new chunk arrives, the robot may have advanced several control steps since the observation that produced it → early actions are stale, remaining actions may not align smoothly with the buffered trajectory.

**Latency-aware linear-overlap buffer (lines 509–528):** let `k` = number of control steps executed since the previous chunk was integrated (estimates inference delay); `k_max` = maximum number of old actions to discard. When new chunk `A_new` arrives, buffer first **removes its leading `min(k, k_max)` actions** (timesteps the robot has likely already passed). After trim, `a_i^old` and `a_i^new` index the same absolute control timestep. Remaining actions merged with current buffer `A_old` over **overlap window `L = min(|A_old|, |A_new|)`**, so `L` never exceeds horizon `H`:

> **Eq 1 (verbatim, line 523):**  `a_blend_i = w_i · a_i^old + (1 − w_i) · a_i^new`,   `w_i = 1 − i/(L−1)`,   where `i = 0, …, L−1`.

Weighting starts from the buffered trajectory and gradually shifts to the new prediction, reducing discontinuities at chunk boundaries. Actions of `A_new` beyond the overlap window are **appended unchanged**. Figure 5(a) (lines 551–577): leading disposed actions trimmed → overlap window fused by linear blend → fresh tail extends trajectory.

### 5.3 Temporal ensembling, ACT-style (lines 530–592)

Following **Zhao et al. (2023a)**, smooths async chunks by aggregating predictions **in absolute time**. Instead of replacing previous predictions, each action is assigned to its target control timestep and all predictions for the same timestep are retained. Let `P(t) = {(j, a)}` denote predictions associated with timestep `t`, where `j` indexes the inference call and smaller `j` = earlier call. Executed action:

> **Eq 2 (verbatim, lines 582–584):**  `ā_t = [ Σ_{(j,a)∈P(t)} e^{−m·ρ_t(j)} · a ] / [ Σ_{(j,a)∈P(t)} e^{−m·ρ_t(j)} ]`

where `ρ_t(j)` = local position of call `j` within `P(t)`, counted from 0 for the oldest call, increasing toward the newest. **Earlier predictions receive larger weights**; later predictions exponentially discounted. This up-weighting of older predictions is "inherited from ACT's temporal-ensembling default rather than a design choice specific to EVA-Client." Coefficient `m` controls decay strength; **default `m = 0.01` → nearly uniform averaging.** Predictions beyond a fixed temporal horizon are pruned for efficiency. Each chunk may cover multiple future timesteps → background worker integrates the **full, uncropped chunk** to maximize temporal overlap. Figure 5(b): every chunk active at an ensemble slice contributes to the executed action.

### 5.4 Naive asynchronous replacement (lines 594–597) — ablation baseline

Minimal baseline: most recent chunk **simply replaces the buffer**, indexed by a global timestep so the executed action corresponds to elapsed time since the chunk was produced → **compensates for latency by indexing, not by blending.** Useful as a control to **isolate the contribution of smoothing.** (This is the `naive-async ablation baseline` keyword.)

### 5.5 Real-Time Chunking, RTC (lines 599–615)

Reduces seams **at the source rather than post-hoc** (Physical Intelligence, 2025). Each inference request carries the **previously committed action chunk back to the server**, shifted forward by the latency `k`, with trailing entries padded with the last committed action. Generation thereby **conditioned to remain consistent with what the robot is already executing.**

**Deliberate division of labor:** client only does the latency shift + attaches prior actions to the request; **server owns the delta conversion, normalization, conditioned generation.** Returned chunks still pass through the **linear-overlap buffer for a final smoothing pass** ("double smoothing"). Inexpensive: when server conditioning succeeds, successive chunks already agree over the overlap window → linear blend is **near-identity**, acts only on residual boundary disagreement. Conceptually: RTC moves **part of the smoothing burden into the model**; client retains a robust fallback.

### 5.6 Debuggability (lines 617–619)

Each strategy fully logged through the **three action streams** of §8 — effect of a smoothing method, including latency artifacts, can be inspected after the run rather than guessed at on the robot.

---

## 6. Data collection (§7, lines 622–649)

The §5 operation model drives a trained policy, but the same console + robot descriptions + transport backends also support the inverse problem — **producing the demonstrations a policy is trained on.** Collect mode records teleoperated episodes directly into a training-ready dataset. **Unlike other modes, Collect runs no policy at all** — a human drives, the client only records.

- **Teleoperated recording (lines 629–635):** operator drives by teleoperation (leader arm, follower mirrors); each frame recorded as a **synchronized pair** — robot's measured state + commanded action, each available as joint angles and (via FK) end-effector pose. Recording gated behind an **explicit activation step** (capture never begins accidentally). Because both state and action stored, a recording is simultaneously replayable for review AND usable as a supervised training target. Live capture on ROS1/ROS2/ZMQ transports; a config becomes a collection config by **declaring a recording schema** — that activates the Collect tab.
- **Training-ready output (lines 637–642):** episodes in **LeRobot format** (Cadene et al., 2026) — same on-disk layout EVA-Client reads back for offline replay (§4). One dataset per task: per-step observations + actions in a columnar table, one **H.264 video per camera**, dataset metadata. To satisfy the format's fixed-rate assumption, **per-frame timestamps synthesized at exact intervals** while jittery real capture time preserved in a separate field. Saving handed to a **background writer** so live capture never stalls; recordings **appended, not overwritten.**
- **Quality control (lines 644–649):** every finished episode checked frame-by-frame for **non-monotonic timestamps, missing/malformed camera frames, wrong-dimension vectors, non-finite values, video/table length mismatches** → flagged clean or problematic. **Flagged episodes never silently discarded:** offending fields **zero-filled** to keep the table regular, issue recorded for review — bad data quarantined, not lost. Operator reviews by replaying open-loop in the console (no robot/policy server required), records a **pass/fail verdict + optional free-text note, written in place without touching the recorded trajectory.**

---

## 7. Evaluation and logging (§8, lines 652–699)

"Deployment is the least consolidated part of the VLA stack; **evaluation is the least systematic.**" Teams judge a new checkpoint by running it a few times and forming an impression — subjective, unrecorded, impossible to reproduce/audit. EVA-Client treats evaluation + logging as a **core subsystem**, so "is this checkpoint better?" becomes a recorded, comparable measurement.

- **Why hard on real robots (lines 665–669):** unlike simulation, a physical evaluation **cannot be replayed.** Scene drifts between trials; a single overall impression is subjective; the most informative signal (what the policy actually saw and did) is lost unless captured. A credible harness must **standardize the trial protocol, score each trial against explicit milestones** (not a single overall judgment), and persist enough info to reconstruct/compare runs after the fact.
- **Scored trials and scenes (lines 671–678):** evaluation organized into **scenes** — each pairing an object configuration + a prompt + a per-trial list of **milestone outcomes.** Each scene run for a configured number of trials; every trial scored against its milestones → **graded process score, not just binary success** (+ optional free-text note). Every trial persisted as a structured record (one per trial), keyed by `(scene, position, trial)`, carrying milestone outcomes, graded score, duration, free-text note. Each record **bound to a recorded video** via a clip identifier minted when the run starts — video and score stay linked across restarts. Per-scene and per-checkpoint statistics (success rate, process score) computed **from the logs**, not reduced to a single anecdotal outcome.
- **Multi-checkpoint comparison (lines 680–683):** several checkpoints evaluated within one session. Switching the "active model" transparently **re-targets the policy endpoint and re-runs warmup**, so each checkpoint scored on the same scenes/milestones under identical conditions. Results logged per checkpoint, kept separate → lined up side-by-side in the result viewer.
- **Three action streams (lines 685–691):** for every run, three parallel timestamped action streams — **(1) raw per-chunk predictions** returned by the policy, **(2) smoothed actions** after the inference strategy's buffering, **(3) executed actions** actually sent to the robot. Each step tagged with its **source chunk.** Recording all three (not only the final command) makes the pipeline debuggable — plotting together reveals **chunk boundaries, overlap regions, latency compensation** directly, attributes a bad motion to **model vs smoothing vs execution layer.** On the executed stream, the narrow transition zone where two chunks overlap is visible directly.
- **Read-only result viewer (lines 693–699):** logged results explorable through a read-only web viewer that connects to **neither robot nor policy server.** Renders per-checkpoint statistics, per-scene breakdowns, **side-by-side comparison across checkpoints**, each scored trial linking to its recorded video + milestone outcomes. Three action streams exported as **per-run tables organized by prompt/mode/strategy**, chunk ownership tracked per step → overlap regions visualizable. Viewer consumes only persisted logs → results inspectable/shareable/re-analyzed long after the robot is powered down. **Closes the loop from physical run to an auditable, reproducible record.**

---

## 8. Limitations and roadmap (§9, lines 702–731)

**Limitations (lines 703–707):**
- Deployment infrastructure, **not a policy or a benchmark** — trains no models; qualitative task outcomes are illustrative, **not a controlled study.**
- Live transport covers ROS1/ROS2/ZMQ, but **cameras on non-ROS robots** still depend on a middleware-specific image source.
- IK solver targets **serial-arm manipulators**; other morphologies supported only through their own robot descriptions.

**Roadmap (lines 709–731):**
1. **RL data collection** — close the loop from evaluation back to external training; treat policy rollouts + reward/outcome labels as material for RL / interactive fine-tuning. Human-in-the-loop: operator takes over mid-rollout, correction recorded as targeted training signal.
2. **Agentic policies** — runtime for hierarchical policies (high-level planner/VLA dispatches sub-goals to low-level controllers); narrow interfaces already let the same client host controllers while exposing planner hooks.
3. **Data annotation** — extend Collect with fine-grained task/sub-task annotation; segment long-horizon episodes into labeled sub-task units + milestones within the same LeRobot dataset.
4. **Broader robot morphologies** — widen supported set to humanoid + mobile-base by adding the descriptions + kinematic models each requires, registered through the same interface.

> "As with our current collection modes, we release the tooling that produces and labels the data, **not the data itself.**" (line 731)

---

## 9. Strengths / Limitations / Verdict

**Strengths**
- **Genuine category gap filled.** Training side consolidated (openpi/LeRobot/StarVLA/VLA Foundry); deployment side was per-policy/per-robot scripts. EVA-Client is the first open client consolidating the whole real-robot iteration loop. Architecture is the contribution, and it is clearly decomposed.
- **Clean orthogonal-decoupling design.** Signal-source × transport × robot × inference-strategy as an independent grid with narrow dataclass interfaces is the right abstraction; "adding a robot is a configuration choice, not a coding task" is verifiable from the layer description.
- **Unified inference-strategy surface.** Five strategies (sync / async-linear-overlap / ACT-ensemble / naive-async / RTC) behind one config, with the two formal blending rules (Eq 1, Eq 2) transcribed exactly — directly comparable, which the paper notes is the point (they are "usually implemented in isolation and tied to one model").
- **Honest scope.** Explicit "not a policy/benchmark/dataset," "illustrative observations not a controlled study," serial-arm-only IK — does not overclaim.
- **Three-action-stream logging** (raw/smoothed/executed) is the most diagnostically valuable idea — makes the inference pipeline debuggable and attributes failure to model vs smoothing vs execution.

**Limitations** (paper-stated + observed)
- **No quantitative evaluation.** Zero benchmark tables; the table-tennis/cloth-fold outcomes are illustrative filmstrips. No success-rate numbers, no ablation of strategies under matched conditions, no latency/throughput measurements. The paper is candid about this ("ongoing development report"), but it means the central soft claim — that strategy choice "can determine whether a task succeeds or stalls" — is demonstrated anecdotally, not measured.
- **Naive-async is the only ablation.** The paper positions naive-async as "a control to isolate the contribution of smoothing" but never reports the side-by-side comparison it implies; the ablation is named, not run.
- **IK discontinuities hand-waved.** "Near singularities or joint limits discontinuities can still occur and are logged" — no frequency/impact data.
- **Single reference robot (dual-arm Piper, 14-D)**; the other 5 platforms are listed as supported but the paper does not characterize parity of support.
- **ROS-coupled camera limitation** for non-ROS robots is open.

**Verdict:** A well-motivated **systems/infrastructure** paper — the deployment counterpart to mature training frameworks. Value is the architecture (5 layers, orthogonal grid), the unified inference-strategy surface with two exact blending equations, and the three-action-stream logging substrate. The absence of any controlled quantitative evaluation is the dominant caveat: every empirical claim is illustrative. Cite for the deployment-loop abstraction and the smoothing-strategy formalization; do not cite for benchmark numbers (there are none). Most reusable artifacts: Eq 1 (linear-overlap blend), Eq 2 (ACT ensemble), the PyRoki continuous-IK recipe (LM + trust-region + dense-Cholesky + warm-start, three weighted costs), and the three-stream logging contract.

---

## 10. Internal-consistency check (source-first)

Cross-checked every prose claim against source — all reconcile; **no prose-vs-table numeric inconsistencies** (there are no numeric tables to contradict):
- "14-D action vector = 2 × (6 joints + 1 gripper)" ↔ line 344 ✓
- Five execution modes (Table 1) ↔ lines 203–216, §5 prose lines 399–414 ✓
- Five inference strategies (Table 2) ↔ §5 prose enumeration + Async/Smoothing columns ✓
- Eq 1 weights `w_i = 1 − i/(L−1)` start at buffered trajectory (`w_0=1`), shift to new prediction (`w_{L−1}=0`) ↔ line 525 ✓
- Eq 2 default `m = 0.01` → "nearly uniform averaging" ↔ line 589 ✓
- Three action streams (raw/smoothed/executed) ↔ §8 lines 686–688 ✓
- Six supported platforms (Franka, UR5e, Galaxea R1-lite, AgileX Piper, AgiBot G2, ARX R5) ↔ line 345 ✓

**One mild framing tension (flagged, not a contradiction):** the abstract/§1 positions async/RTC/temporal-ensembling as comparable strategies, but §6's only worked deployment evidence (Figure 4) shows async *beating* sync on table tennis — sync is dismissed as inadequate rather than shown as a legitimate alternative under matched measurement. Consistent with the "illustrative, not controlled" caveat; readers should not read Figure 4 as a benchmark.

---

*Breakdown built source-first from `paper_layout.txt` (867 lines). Tables 1–2, Eq 1–2, the five-layer architecture, the continuous-IK recipe, and the three-action-stream logging contract transcribed verbatim with sourcing line-ranges. Figure-derived task outcomes (Figures 3–5: console screenshots, table-tennis/cloth-fold filmstrips, smoothing timelines) are flagged as illustrative, not back-filled, per the established repo rule that figure-derived sections are the weak spot — and doubly so here since the paper itself declares them uncontrolled.*
