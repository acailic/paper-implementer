# Writeup — GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

There's a debate in the agent world that everyone has an opinion on but nobody has properly tested: are computer-use agents better off operating through graphical interfaces (clicking, typing, dragging on a desktop) or through programmatic skills (CLI commands that manipulate application state)?

Every benchmark to date has compared these two modalities with different tasks, different starting conditions, different success criteria, and different rules about what the agent is allowed to do. That's not a comparison — that's two separate experiments pretending to be one.

This paper fixes that. They build 440 desktop tasks across 18 real applications — Audacity, LibreOffice, GIMP, draw.io, Chrome, Zoom, FreeCAD, you name it — and run them through both a screen-only GUI agent and a skill-mediated CLI agent with the exact same task description, the exact same starting state, and the exact same final-state verifier. The only thing that changes is the action interface. Screenshots and clicks vs CLI-Anything skills.

## The headline numbers

The strongest GUI agent (GPT-5.4) gets **59.1%** full pass rate. The strongest CLI agent with original skills (Codex GPT-5.5) gets **48.2%**. GPT-5.4 is the weaker model but wins through the GUI. That's already interesting — the interface can compensate for model limitations.

But here's the real punchline. The authors then ask: how much of that CLI gap is because the skill layer is just incomplete? They systematically audit every CLI-Anything skill against the verifier checkpoints. The answer: **only 37.6% of verifier checkpoints can be satisfied by the original skills**. Sixty-some percent of what the task requires simply isn't exposed through the CLI skill interface.

So they run a diagnostic where they patch the skills (using verifier information — this is explicitly a diagnostic upper bound, not a deployable solution). CLI jumps from 48.2% to **69.3%**. Now it beats the best GUI agent. The CLI wasn't weaker — its tools were incomplete.

## What I found most surprising

The per-workflow breakdown flips several intuitions. I assumed GUI would dominate visual tasks like GIMP and draw.io because those are "visual" applications. Wrong. CLI is competitive or stronger in Visual Design because draw.io tasks are really about structured artifacts — pages, shapes, labels, connectors — and those map cleanly to programmatic operations. The CLI agent can say "add a shape called UserService on page 2" directly, while the GUI agent has to visually navigate a canvas, place shapes, type labels, draw connectors, and track state across multiple pages without losing anything.

Where GUI genuinely dominates: Audio (Audacity label tracks), Presentations (LibreOffice Impress slide manipulation), Communication (Zoom settings). These are cases where the application interface *is* the workflow — the menu structure and visible controls directly expose the steps you need. The GUI agent sees the path; the CLI agent has to reconstruct it from incomplete skill documentation.

The second surprise was the procedural grounding experiment. When you give GUI agents explicit step-by-step instructions ("click Tracks > Add New > Mono Track, then click the dropdown arrow, choose Name..."), full pass rate barely moves — from 59.7% to 60.2%. But average execution time drops 20%. The agent stops wasting steps on exploration. But it still fails at the same rate because its actual bottleneck is visual grounding — reliably clicking the right thing, tracking state across long sequences, not giving up early.

## The failure modes are completely different

CLI agents fail because:
- The skill layer doesn't expose the operation they need (skill coverage gap)
- They have to guess defaults that GUI users inherit automatically — object naming conventions, internal identifiers, label vs name distinctions (implicit default reconstruction)
- Critical application state isn't visible through any skill command, so they hallucinate plausible-but-wrong configurations (unobservable semantics)

GUI agents fail because:
- They can't find the right control — menus, tabs, dialogs, hidden settings. They click around searching for the right path and run out of steps.
- They get the workflow wrong — wrong order of operations, missing confirmation dialogs, stopping too early.
- They declare success without checking. They go through a plausible sequence of actions and say DONE, but the exported file doesn't exist or the saved state didn't actually change.

These are genuinely complementary. CLI's bottleneck is the *breadth and accuracy of its tool interface*. GUI's bottleneck is the *reliability of its perceptual-motor execution chain*.

## What this means for people building agents

Three takeaways I'd carry away:

**Skill coverage is the central scaling problem for CLI agents.** 37.6% coverage is not a minor gap — it's the dominant explanation for CLI underperformance. If you're building a skill-mediated agent, the quality and coverage of your skill layer matters more than the choice of underlying model. The paper doesn't solve this (building skills automatically at high coverage remains open) but it quantifies exactly how much is at stake.

**Modality choice should be workflow-dependent, not app-category-dependent.** "Visual" apps don't automatically favor GUI. What matters is whether the interface exposes the intended workflow directly (GUI wins) or whether the target state is a structured artifact (CLI wins). An adaptive router that picks modality per task would outperform either alone.

**The GUI-CLI question is really about where execution logic lives.** In GUI, the application interface encodes the workflow — the agent discovers it. In CLI, the skill layer encodes the workflow — the agent invokes it. Neither is inherently better. The real design question is: where should the executable task structure be engineered? Into visible workflows, verified skill interfaces, or hybrid environments that combine both?

## References
- Paper: https://arxiv.org/abs/2606.24551
- Official code: https://github.com/rebeccaz4/gui-vs-cli
- Breakdown: `breakdown.md`
