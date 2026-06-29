# Writeup — OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning

> Your own explanation of the paper, as if teaching it to a peer who hasn't
> read it.

## The one-paragraph version

OPID is a training framework that makes RL-trained agentic LLMs learn faster
by extracting hindsight "skills" from the agent's own completed trajectories.
After each rollout, an LLM analyzer summarizes what went right or wrong at two
levels — a global episode-level workflow and precise step-level guidance at
critical decision points. These skills are then injected into the trajectory's
context, and the policy re-scores its own responses with and without the skill.
The resulting per-token log-probability shift becomes a dense self-distillation
signal that's combined with the standard GRPO outcome advantage. At inference
time, none of this machinery is needed — the skills have been distilled into
the model weights.

## The problem

When you train an LLM agent with reinforcement learning on multi-step tasks
(embodied household tasks, web shopping, search-augmented QA), the reward is
typically binary and delayed — you only know if the whole trajectory succeeded
or failed. This is especially painful in long-horizon settings where a single
mistake early on cascades into failure 20 steps later. GRPO handles this by
normalizing rewards within groups, but every token in a trajectory still gets
the same scalar advantage. The agent has no idea *which* specific action at
*which* step was the mistake.

Previous work tried to fix this with skill-conditioned distillation — giving
the model extra natural-language guidance during training. But those methods
need external skill libraries that someone has to curate, and retrieved skills
can be stale or mismatched with what the current policy actually encounters.

## The idea

The authors' observation is simple and elegant: the agent's own completed
trajectories already contain all the decision knowledge you need. A successful
trajectory demonstrates a valid workflow; a failed trajectory demonstrates
what to avoid. You just need to extract that knowledge as natural-language
skills and use it as a training signal. Since these skills come from the current
policy's own rollouts, they're guaranteed to match the policy's state
distribution — no retrieval mismatch possible.

## How it works (the intuition)

Think of it as a post-game analysis. After an agent finishes a task, you sit it
down (well, an external analyzer model) and ask: "What was the overall strategy
here? And at which critical moments did things go right or wrong?" The analyzer
produces two kinds of skills:

1. **Episode-level skill**: The big picture. For a success, it's something like
   "First locate the object, then clean it at the sink, then place it." For a
   failure, it's "You kept trying to put a dirty object in the cart without
   cleaning it first."

2. **Step-level skills**: Precise guidance at 2-5 critical moments. Like
   "At step 0, go directly to the countertop where the kettle is" or "At step
   2, check if the soapbar needs cleaning before moving it."

Now here's the clever part. At each timestep, OPID picks the *most appropriate*
skill — step-level at critical moments, episode-level everywhere else. This
critical-first routing means you get precise guidance where it matters and
broad guidance where precision isn't needed.

To turn these skills into a learning signal, OPID injects the selected skill
into the interaction history and asks the policy to re-score its own generated
response. If the skill-augmented context makes a token more likely, that token
gets a positive advantage (it's aligned with hindsight wisdom). If less likely,
it gets a negative advantage (it contradicts what the trajectory analysis says
should have happened).

This token-level advantage is then added to the standard GRPO trajectory-level
advantage. The episode advantage tells the policy "this trajectory was good/bad
overall" while the skill advantage tells it "at this specific token, here's
what hindsight says you should have done."

Because the skill signal comes from a paired scoring pass (same policy, same
response, different context), it's guaranteed to be on-policy and
distribution-matched. And at inference time, none of this is needed — the
policy has internalized the skill knowledge through training.

## What I learned by implementing it

The most important implementation detail is the analyzer prompt. The paper
uses a carefully structured JSON prompt that asks for exactly three fields:
episode_summary, episode_skill, and step_skills (a dict of step indices to
skills). The step indices are 0-based and must match the trajectory steps.
Getting this prompt right is probably half the battle.

The paired scoring is also non-trivial. You need to do a full forward pass
through the policy twice per trajectory step — once with the original context
and once with the skill-augmented context. This doubles the compute per step
but only during training. The old policy parameters are frozen during scoring,
so the skill advantage is fully detached.

λ_skill = 0.001 is surprisingly small. The skill advantage is a gentle nudge,
not a dominant force. This makes sense — you want the RL outcome signal to be
primary, with skills providing fine-grained shaping.

## What surprised me / was harder than expected

The biggest surprise is how much OPID beats Skill-GRPO. Skill-GRPO uses
external skills during training but removes them at inference, and the result
is often *worse* than plain GRPO. On ALFWorld Qwen2.5-3B, Skill-GRPO without
inference skills scores 60.2 while plain GRPO scores 75.0. The train-test
mismatch is devastating. OPID's approach of distilling skills into the model
parameters avoids this entirely.

The theoretical appendix is unusually rigorous. The proof that the skill loss
is a "relative-KL" (reverse-KL minus behavior-KL) and the characterization
of when it exactly recovers reverse-KL (β = λ_skill) is clean and
non-trivial. The analysis of critical-first routing as oracle approximation
under a specialization assumption is also elegant.

The sample efficiency result is striking: OPID at 60% data matches GRPO at
100% data. This suggests that dense token-level supervision extracts
significantly more learning signal per rollout than sparse outcome rewards.

## References

- Paper: https://arxiv.org/abs/2606.26790
- Code: https://github.com/jinyangwu/OPID
- Breakdown: `breakdown.md`
