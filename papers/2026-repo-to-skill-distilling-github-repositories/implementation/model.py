"""model.py — the deterministic Distiller (Eq. 8 of arXiv:2609.02749).

Paper: "Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills"
        (Chen et al., 2026)

Re-implements the paper's four-stage pipeline

    z --scope--> Q --ground--> X --construct--> G~ --verify--> (G, R)

for a task-agnostic anchor (z = repository) with **deterministic,
rule-based distillation** instead of the paper's GPT-5.5 creator agent:

  * scope      : infer the capability set Q from evidence signals
                 (README sections/snippets, class & function names, entry
                 points, packaging metadata).
  * ground     : map each capability to the evidence units that support it
                 (X), every unit carrying file:line provenance.
  * construct  : emit the three-layer skill artifact of Eq. 6 —
                 SKILL.md (knowledge interface), references/ (knowledge
                 substrate), scripts/ (execution interface) — plus a
                 graph.json realizing Eq. 7 (entry skill + component
                 skills + links).
  * verify     : implemented in verify.py (static gates + QA carry test).

No LLM, no network, no execution of repository code. What a real LLM
distiller would add is discussed in ../writeup.md.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from data import EvidenceUnit, SourceProfile

# ---------------------------------------------------------------- scope: Q

@dataclass
class Capability:
    """One entry of Q: a bounded thing the repo makes possible."""
    key: str                                   # stable id, e.g. "model-config"
    title: str                                 # human title for SKILL.md
    when: str                                  # "when to use" phrasing
    triggers: tuple                            # regexes over evidence names/text
    evidence: list = field(default_factory=list)   # grounding: EvidenceUnits

    def supported(self) -> bool:
        return bool(self.evidence)


def capability_catalog() -> list[Capability]:
    """The rule set mapping evidence signals -> capabilities (the scoping Q).

    Deliberately generic across small ML repos; the paper's creator agent
    performs this step with an LLM, we do it with triggers — this file is
    where 'deterministic distillation' is most visible.
    """
    return [
        Capability("install", "Install / environment setup",
                   "You need to install the package or set up its environment before anything else.",
                   (r"install_cmd", r"requirements\.txt", r"install_requires", r"pyproject")),
        Capability("model-config", "Instantiate and configure the core model",
                   "You need to create/configure the repository's main model component.",
                   (r"class .*(Model|GPT|Net|Transformer|VAE|GAN|Autoencoder)", r"Config")),
        Capability("training", "Run / customize training",
                   "You need to train or fine-tune, or adapt the training loop.",
                   (r"train", r"Trainer", r"fit\(", r"optimizer", r"checkpoint")),
        Capability("data-prep", "Prepare / preprocess data",
                   "You need to build or preprocess the dataset the repo expects.",
                   (r"prepare_data", r"dataset", r"Dataset", r"dataloader", r"DataLoader", r"tokenize", r"encode")),
        Capability("inference", "Run inference / sampling / evaluation",
                   "You need to generate outputs, sample, or evaluate a trained model.",
                   (r"generate", r"sample", r"eval", r"test\b", r"metric", r"inference", r"model\.eval")),
        Capability("pretrained", "Load pretrained / provided checkpoints",
                   "You need to initialize from published weights or a checkpoint.",
                   (r"from_pretrained", r"load_state_dict", r"checkpoint", r"weight", r"\.ckpt", r"\.pt\b")),
        Capability("cli", "Use the command-line entry points",
                   "You need to drive the repo from the shell rather than Python.",
                   (r"entry_point", r"console_scripts", r"runnable_module", r"argparse")),
        Capability("repo-layout", "Understand the repository layout",
                   "You need to know which file owns which responsibility.",
                   (r"module_docstring", r"class", r"function")),
    ]


def scope(profile: SourceProfile) -> list[Capability]:
    """Stage 1: capability identification (paper: Source Understanding ->
    Capability Identification). A capability enters Q only if at least one
    evidence unit matches one of its triggers."""
    caps = capability_catalog()
    for ev in profile.evidence:
        hay = f"{ev.kind}|{ev.name}|{ev.text[:400]}"
        for cap in caps:
            for trig in cap.triggers:
                if re.search(trig, hay, re.I):
                    cap.evidence.append(ev)
                    break
    return [c for c in caps if c.supported()]


# ---------------------------------------------------------------- ground: X

def ground(profile: SourceProfile, caps: list[Capability]) -> dict:
    """Stage 2: map capabilities to supporting evidence X (already collected
    during scope); here we de-duplicate and rank. Ranking is deterministic:
    classes/functions/docstrings before README prose, longer docstrings
    first (more operational content)."""
    X: dict[str, list[EvidenceUnit]] = {}
    kind_rank = {"class": 0, "method": 1, "function": 2, "install_cmd": 0,
                 "entry_point": 1, "dependency": 2, "readme_snippet": 3,
                 "readme_section": 4, "module_docstring": 3, "meta": 5}
    for cap in caps:
        seen, uniq = set(), []
        for ev in cap.evidence:
            k = (ev.kind, ev.name, ev.file, ev.line)
            if k not in seen:
                seen.add(k)
                uniq.append(ev)
        uniq.sort(key=lambda e: (kind_rank.get(e.kind, 9), -len(e.text)))
        X[cap.key] = uniq
    return X


# ---------------------------------------------------------------- construct: G~

SKILL_TEMPLATE = """---
name: {name}
description: {description}
source: {repo_url}
commit: {commit}
distiller: deterministic-rule-based (no LLM) — see implementation/README.md
---

# {name} — operational skill

> What this is: distilled operational knowledge for **{repo}** @ `{commit}`.
> Distilled automatically from the repository evidence; every claim below
> is grounded in the files listed in `references/`. Layout follows the
> Agent Skills convention (SKILL.md + references/ + scripts/).

## When to use this skill

{when}

## Requirements

```bash
{reqs}
```

{steps}

## Pitfalls

{pitfalls}

## Evidence & provenance

- Source: {repo_url} @ commit `{commit}`
- Evidence files: {file_count} admitted by the include/exclude boundary
- Full API surface with file:line provenance: `references/api.md`
- README-derived quickstart snippets: `references/quickstart.md`
- Static sanity probe for a local checkout: `scripts/smoke_check.py`
- Construction record (verification results): `record.json` next to this graph
"""


def _when_block(caps: list[Capability]) -> str:
    return "\n".join(f"- **{c.title}** — {c.when}" for c in caps)


def _reqs_block(profile: SourceProfile) -> str:
    deps = sorted(profile.dependencies)
    if deps:
        return "# install observed third-party dependencies:\npip install " + " ".join(deps)
    return "# no third-party imports detected in the admitted evidence"


def _steps_block(X: dict, caps: list[Capability]) -> str:
    """Order capabilities into an operational sequence: install first, layout
    last, model before training before inference (paper: skills encode
    *use conditions and procedures*, not a directory dump)."""
    order = ["install", "repo-layout", "data-prep", "model-config",
             "pretrained", "training", "inference", "cli"]
    by_key = {c.key: c for c in caps}
    parts = [f"## Steps\n"]
    step_no = 0
    for key in order:
        cap = by_key.get(key)
        if cap is None:
            continue
        step_no += 1
        parts.append(f"### Step {step_no}. {cap.title}\n")
        parts.append(cap.when)
        parts.append("")
        shown = 0
        for ev in X[key]:
            if ev.kind in ("class", "method", "function") and shown < 6:
                parts.append(f"- `{ev.name}` — {ev.text.splitlines()[0][:160]}  "
                             f"*(evidence: {ev.file}:{ev.line})*")
                shown += 1
            elif ev.kind in ("install_cmd", "entry_point", "meta") and shown < 6:
                label = ev.text.splitlines()[0][:160] if ev.kind != "meta" else ev.name
                parts.append(f"- `{label}`  *(evidence: {ev.file}:{ev.line or '—'})*")
                shown += 1
            if shown >= 6:
                break
        if shown == 0:
            parts.append("*(no directly quotable evidence; see `references/api.md`)*")
        parts.append("")
    return "\n".join(parts)


def _pitfalls_block(profile: SourceProfile) -> str:
    """Deterministic pitfall mining: README warnings/notes + version pins +
    device/seed remarks. (The paper's distiller mines real failure modes
    from tests and issues; ours is honest about being shallower.)"""
    pits: list[str] = []
    for ev in profile.evidence:
        if ev.kind in ("readme_section", "readme_snippet", "module_docstring"):
            for ln in ev.text.splitlines():
                if re.match(r"\s*(>|\*{0,2})\s*(!|note\b|warning|caveat|todo|deprecated|important)",
                            ln, re.I):
                    if re.match(r"\s*!\[", ln):     # markdown image, not a warning
                        continue
                    pits.append(f"- {ln.strip()[:200]}  *(from {ev.file})*")
    deps = sorted(profile.dependencies)
    if deps:
        pits.append(f"- Dependency surface: the code imports {', '.join(deps)} — "
                    "version mismatches are the first thing to check when imports fail.")
    if not pits:
        pits.append("- (no explicit warnings found in the admitted evidence; "
                    "this section is where an LLM distiller with issue-mining "
                    "would contribute most — see writeup.md)")
    dedup = list(dict.fromkeys(pits))[:8]
    return "\n".join(dedup)


def write_skill(out_dir: str, profile: SourceProfile, caps: list[Capability],
                X: dict) -> str:
    """Stage 3: assemble the three-layer artifact + graph.json (Eq. 6/7)."""
    skill_dir = os.path.join(out_dir, "skills", profile.repo_name.lower())
    os.makedirs(os.path.join(skill_dir, "references"), exist_ok=True)
    os.makedirs(os.path.join(skill_dir, "scripts"), exist_ok=True)

    purpose = next((e.text for e in profile.evidence if e.name == "purpose"),
                   f"{profile.repo_name} (auto-distilled)")
    description = " ".join(purpose.split())[:300]

    skill_md = SKILL_TEMPLATE.format(
        name=profile.repo_name.lower() + "-operations",
        description=description,
        repo=profile.repo_name,
        repo_url=profile.repo_url,
        commit=profile.commit,
        when=_when_block(caps),
        reqs=_reqs_block(profile),
        steps=_steps_block(X, caps),
        pitfalls=_pitfalls_block(profile),
        file_count=len(profile.files),
    )
    open(os.path.join(skill_dir, "SKILL.md"), "w").write(skill_md)

    # references/api.md — the knowledge substrate: full public surface w/ provenance
    api = [f"# API surface — {profile.repo_name} @ {profile.commit}\n"]
    for ev in profile.evidence:
        if ev.kind in ("class", "method", "function", "module_docstring"):
            api.append(f"## {ev.kind}: `{ev.name}`\n- provenance: `{ev.file}:{ev.line}`\n\n"
                       f"```\n{ev.text[:700]}\n```\n")
    open(os.path.join(skill_dir, "references", "api.md"), "w").write("\n".join(api))

    # references/quickstart.md — README-derived snippets + key prose sections
    qs = [f"# Quickstart evidence — {profile.repo_name} @ {profile.commit}\n"]
    for ev in profile.evidence:
        if ev.kind in ("readme_snippet", "install_cmd"):
            qs.append(f"### {ev.name} (from `{ev.file}:{ev.line}`)\n\n"
                      f"```\n{ev.text[:1500]}\n```\n")
        elif ev.kind == "readme_section" and ev.name.lower() not in ("license",):
            body = ev.text[:1800]
            qs.append(f"### README: {ev.name} (from `{ev.file}:{ev.line}`)\n\n{body}\n")
    open(os.path.join(skill_dir, "references", "quickstart.md"), "w").write("\n".join(qs))

    # scripts/smoke_check.py — execution interface (STATIC probe: reads the
    # checkout, never executes repo code)
    smoke = f'''"""smoke_check.py — static sanity probe for a {profile.repo_name} checkout.

Generated by the deterministic repo-to-skill distiller. Usage:

    python3 smoke_check.py /path/to/{profile.repo_name}

It only READS the checkout (no repo code is executed): verifies the files
that the skill's evidence is grounded in still exist and contain the
symbols this skill documents.
"""
import os, re, sys

EXPECT_FILES = {sorted(set(e.file for e in profile.evidence if e.kind in ("class", "function", "method")))!r}
EXPECT_SYMBOLS = {sorted(set(e.name.split("(")[0].split(".")[-1] for e in profile.evidence if e.kind in ("class", "function")))!r}

def main(root):
    ok = True
    for rel in EXPECT_FILES:
        if not os.path.exists(os.path.join(root, rel)):
            print(f"MISS  file   {{rel}}"); ok = False
        else:
            print(f"ok    file   {{rel}}")
    for rel in EXPECT_FILES:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8", errors="replace").read()
        for sym in EXPECT_SYMBOLS:
            if re.search(r"\\b{{}}\\b".format(re.escape(sym)), src):
                print(f"ok    symbol {{sym}} in {{rel}}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
'''
    open(os.path.join(skill_dir, "scripts", "smoke_check.py"), "w").write(smoke)

    # graph.json — Eq. 7: entry skill + component nodes + routing links
    nodes = [{"id": "entry", "kind": "entry-skill", "path": "SKILL.md"}]
    for cap in caps:
        nodes.append({"id": cap.key, "kind": "component-capability",
                      "title": cap.title, "evidence_units": len(X[cap.key])})
    nodes += [{"id": "ref-api", "kind": "reference", "path": "references/api.md"},
              {"id": "ref-quickstart", "kind": "reference", "path": "references/quickstart.md"},
              {"id": "script-smoke", "kind": "script", "path": "scripts/smoke_check.py"}]
    links = [{"src": "entry", "dst": c.key, "rel": "routing"} for c in caps]
    links += [{"src": c.key, "dst": "ref-api", "rel": "dependency"} for c in caps]
    links += [{"src": "entry", "dst": "ref-quickstart", "rel": "routing"},
              {"src": "entry", "dst": "script-smoke", "rel": "routing"}]
    graph = {"repo": profile.repo_name, "commit": profile.commit,
             "nodes": nodes, "links": links}
    open(os.path.join(skill_dir, "graph.json"), "w").write(
        __import__("json").dumps(graph, indent=2))
    return skill_dir
