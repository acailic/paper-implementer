# Writeup — Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation

> How I'd explain this to a friend over a beer, if they asked "what have you been reading?"

> **Languages:** English (this file) · [Srpski](writeup-sr.md)

---

The simple story goes like this.

You ask a text-to-image model to "draw a scoreboard for the 2026 NBA Finals with both team logos and the series score." The model tries. It doesn't know who played. It doesn't know the score. It doesn't know what the logos look like. It hallucinates something vaguely basketball-shaped and moves on.

This isn't a rendering problem. The model can draw just fine when you give it everything it needs. The problem is that you *didn't give it everything it needs* — and the model has no way to ask for more.

This paper calls that gap the **Context Gap**: the mismatch between what the user says and what the image generator actually needs to do its job. And they build a system that bridges it.

## The idea in one paragraph

Instead of treating the user's prompt as the final input to an image generator, treat it as a *starting point*. An agent looks at the prompt, figures out what's missing, goes out and finds the missing pieces (by reasoning, searching the web, pulling from memory, or self-correcting), assembles everything into a rich detailed prompt, and *then* hands it to the image generator. The generator becomes a renderer at the end of a pipeline, not the whole system.

## The pipeline — three levels of planning, four sources of context

The system has two main parts. **Context-Aware Planning** operates at three levels:

1. **Information-level:** "What don't I know?" — identifies gaps, raises questions, routes them to the right strategy
2. **Content-level:** "Now write the full spec" — assembles all gathered info into a detailed prompt with subject, attributes, layout, style, and text
3. **Generation-level:** "How do I split this across multiple images or turns?" — handles multi-image and multi-turn scenarios

**Context Grounding** fills those gaps from four sources:

| Source | Example |
|--------|---------|
| **Reason** | "CN Tower is the landmark of Toronto" — inferred by the VLM |
| **Search** | "Knicks beat Spurs 4-1 in 2026 NBA Finals" — looked up via Google |
| **Memory** | "The user prefers watercolor style" — remembered from turn 1 |
| **Feedback** | "Only 4 red cars, not 5" — self-checked and corrected |

The whole thing is training-free. It wraps around any existing image generator. In this case they use Qwen-Image-2.0 for generation and GPT-5.5-0424 as the brain doing all the planning and reasoning.

## The benchmark — IA-Bench

The paper also ships a benchmark, which is good because without it you can't really evaluate whether any of this matters. IA-Bench covers four capabilities across 17 subtasks and 730 test instances:

- **Plan** — composition, enumeration, multi-panel layouts, maze paths
- **Reason** — math, science, commonsense, maps, geometry
- **Search** — IP characters (games, movies, anime, celebrities), real-world info (stocks, weather)
- **Memory** — user profiles, conversation history across turns

Evaluation uses VLM judges checking against fine-grained checklists. Two metrics: Pass Rate (all items must pass, strict) and Checklist Accuracy (average proportion satisfied).

## Two things that genuinely surprised me

First — **removing search doesn't just hurt search tasks, it crateres them.** The ablation shows Search PR dropping from 46.1 to 7.8 when you remove the search module. That's not a gradual decline, that's a cliff. It makes sense in hindsight — these tasks literally cannot be solved without external knowledge — but the magnitude is striking. It means search grounding isn't a nice-to-have, it's a load-bearing wall for about a third of real-world image generation tasks.

Second — **the MLLM backbone matters more than I expected.** Swapping GPT-5.5-0424 for Qwen-Plus drops the IA-score from 45.4 to 19.3. That's a 57% collapse from changing just the planner, not the renderer. The whole system's intelligence lives in the planner — it's the one identifying gaps, routing queries, assembling context, writing detailed prompts. When the planner gets weaker, the renderer never even sees good inputs. This is the strongest argument in the paper for why the agentic framing matters: the rendering model barely changed, but the system performance changed enormously because the *context construction* changed.

## What I think about the feedback loop

The honest finding is that feedback adds the least value. Removing it only drops IA-score from 45.4 to 42.1. Two reasons: Qwen-Image-2.0 is already a strong renderer so there's less to correct, and the VLM feedback is generic (not task-specific). The authors acknowledge this openly and suggest future work should push feedback earlier into the pipeline (supervising context-gap identification, not just post-hoc critique). That's the right instinct — waiting until after generation to fix things is inherently limited when the problem was in the prompt.

## The reason vs search boundary

This is a genuinely interesting design problem. Some facts can be solved by the LLM's parametric knowledge ("the CN Tower is in Toronto") and some require web search ("what was the score on August 14, 2025?"). Where do you draw the line? The paper's answer: parametric for commonsense, search for precise facts (exact numbers, dates, names) and dynamic facts (things that change over time). That's clean and principled. But they admit it depends on the MLLM's knowledge boundary — as base models get smarter, more things move from "needs search" to "can reason about it." The boundary is alive.

## What bothers me

**No code.** This is a framework paper with a detailed pipeline description and no implementation. For an agentic system where the engineering details (how exactly do you route questions? how do you prune memory? what does the DAG look like?) matter enormously, this is a real gap. You can't inspect it, can't reproduce it, can't build on it.

**The SOTA numbers are partly a backbone effect.** The system uses GPT-5.5-0424 + Qwen-Image-2.0, both state-of-the-art closed models. The ablation shows that swapping to weaker alternatives collapses performance. So how much of the "45.4 IA-score" is the framework and how much is just having the best tools in the shed? The framework clearly helps (17.4 → 45.4 over bare Qwen-Image-2.0), but the absolute numbers are inflated by proprietary backbone quality.

**Latency.** The full pipeline is way more expensive than one-shot generation. Multiple LLM calls for planning, web API calls for search, VLM calls for feedback, plus the generation itself. DAG execution helps with parallelism but can't eliminate the sequential dependencies. For real-time or cost-sensitive applications, this is a serious constraint.

## The big takeaway

The Context Gap framing is the real contribution here. It gives a name and a structure to something everyone in the field has felt but not formalized: T2I models fail in the real world not because they can't draw, but because they don't know enough about what to draw. The agentic pipeline is a reasonable approach to bridging that gap — and the IA-Bench results prove it works. But until the code is open and the backbone dependency is reduced, it's hard to know how much of this is the framework's design and how much is brute-force model quality.

## References
- Paper: https://arxiv.org/abs/2606.26907
- Breakdown: `breakdown.md`
