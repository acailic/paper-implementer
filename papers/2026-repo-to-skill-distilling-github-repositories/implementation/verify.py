"""verify.py — SkillVerifier: static gates + QA carry test (arXiv:2609.02749).

Paper: "Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills"
        (Chen et al., 2026)

Verification is what "separates distillation from summarization" (Sec. 3.2):
no skill is admitted on the strength of its sources alone, and surviving
gaps are recorded rather than hidden. This module re-implements the two
verification layers that are possible without an LLM:

1. STATIC GATES (paper A.1): required SKILL.md sections, relative-link
   integrity, provenance present (commit + file:line evidence), no
   absolute-local-path leakage, runtime/construction separation, every
   capability grounded in >=1 evidence unit ("no unsupported claims").

2. QA CARRY TEST (miniature of the paper's *recovery experiment*, A.2.2):
   can the skill alone answer test questions about the repository?
   A deterministic retrieval QA scores closed-book (skill files only)
   against open-book (full repo evidence) — the ratio is the fraction of
   operational knowledge the skill *carries* out of the repo.
"""

from __future__ import annotations

import json
import os
import re

from data import SourceProfile

# ---------------------------------------------------------------- static gates

REQUIRED_SECTIONS = ["# ", "## When to use this skill", "## Requirements",
                     "## Steps", "## Pitfalls", "## Evidence & provenance"]
ABS_PATH = re.compile(r"(?<![\w./])(/home/|/Users/|/tmp/|C:\\\\)")


def static_gates(skill_dir: str, profile: SourceProfile, X: dict,
                 caps) -> list[dict]:
    gates: list[dict] = []

    def gate(name, ok, detail=""):
        gates.append({"gate": name, "pass": bool(ok), "detail": detail})

    skill_md = open(os.path.join(skill_dir, "SKILL.md"), encoding="utf-8").read()

    # G1 required sections
    missing = [s for s in REQUIRED_SECTIONS if s not in skill_md]
    gate("required-sections", not missing, f"missing: {missing}" if missing else "all present")

    # G2 link integrity: every relative markdown link / declared path exists
    targets = re.findall(r"`((?:references|scripts)/[\w./-]+)`", skill_md)
    targets += [n["path"] for n in json.load(open(os.path.join(skill_dir, "graph.json")))["nodes"]
                if "path" in n]
    missing_links = [t for t in set(targets)
                     if not os.path.exists(os.path.join(skill_dir, t))]
    gate("link-integrity", not missing_links,
         f"broken: {missing_links}" if missing_links else f"{len(set(targets))} targets resolve")

    # G3 provenance
    g = json.load(open(os.path.join(skill_dir, "graph.json")))
    prov_ok = bool(profile.commit) and profile.commit in skill_md \
        and bool(profile.repo_url) and profile.repo_url in skill_md
    gate("provenance", prov_ok, f"commit={profile.commit} url-recorded={profile.repo_url in skill_md}")

    # G4 no absolute local paths (the paper's local-path-leakage gate)
    leaks = ABS_PATH.findall(skill_md)
    for rel in ("references/api.md", "references/quickstart.md"):
        p = os.path.join(skill_dir, rel)
        if os.path.exists(p):
            leaks += ABS_PATH.findall(open(p, encoding="utf-8").read())
    gate("no-local-path-leakage", not leaks, f"{len(leaks)} leaks" if leaks else "clean")

    # G5 every capability grounded (>=1 evidence unit) = no unsupported claims
    ungrounded = [c.key for c in caps if not X.get(c.key)]
    gate("capabilities-grounded", not ungrounded,
         f"ungrounded: {ungrounded}" if ungrounded
         else f"{len(caps)}/{len(caps)} capabilities have evidence")

    # G6 runtime/construction separation: only runtime files inside the graph
    allowed = {"SKILL.md", "graph.json", "references", "scripts"}
    stray = [f for f in os.listdir(skill_dir) if f not in allowed]
    gate("runtime-separation", not stray, f"stray: {stray}" if stray else "runtime graph clean")

    # G7 scripts are non-empty and statically parseable
    import ast as _ast
    sdir = os.path.join(skill_dir, "scripts")
    sfiles = [f for f in os.listdir(sdir) if f.endswith(".py")] if os.path.isdir(sdir) else []
    parse_ok = True
    for f in sfiles:
        try:
            _ast.parse(open(os.path.join(sdir, f), encoding="utf-8").read())
        except SyntaxError:
            parse_ok = False
    gate("scripts-parseable", bool(sfiles) and parse_ok,
         f"{len(sfiles)} script(s), syntax ok={parse_ok}")

    return gates


# ---------------------------------------------------------------- QA carry test

_STOP = set("a an the of to in on for with and or is are be as it its this that you your "
            "what which how do does using use used from by at into when where who repo "
            "repository code file function class method".split())


def _tokens(text) -> set[str]:
    if isinstance(text, (list, tuple, set)):
        return {t for s in text for t in _tokens(s)}
    return {t for t in re.findall(r"[a-zA-Z_]\w*", str(text).lower())
            if t not in _STOP and len(t) > 1}


def _top_passages(query_tokens: set[str], corpus: list[tuple[str, str]],
                  k: int = 3) -> list[tuple[str, str]]:
    """Deterministic retrieval: rank passages by query-term COVERAGE
    (|q ∩ p| / |q| — how many question terms the passage mentions),
    tie-break by passage focus (fewer tokens). Returns top-k."""
    def key(item: tuple[str, str]):
        path, text = item
        toks = _tokens(text)
        return (-len(query_tokens & toks) / max(1, len(query_tokens)), len(toks))
    return sorted(corpus, key=key)[:k]


def qa_carry_test(skill_dir: str, repo_root: str, profile: SourceProfile,
                  questions_path: str, top_k: int = 3) -> dict:
    """Closed-book vs open-book QA over test questions.

    Deterministic retrieval-QA: for each question, retrieve the top-k
    passages (by query-term coverage) from the corpus, then check what
    fraction of the question's GOLD KEYWORDS appear in the union of the
    retrieved passages. Closed-book corpus = the skill files only;
    open-book corpus = the whole admitted repo evidence. A question is
    'carried' by the skill if it passes closed-book. carry_ratio =
    closed_frac / open_frac — how much of the open-book-answerable
    operational knowledge the skill carries.
    """
    qs = json.load(open(questions_path, encoding="utf-8"))

    # closed-book corpus: passages = paragraphs/blocks of skill files
    skill_corpus: list[tuple[str, str]] = []
    for base, _dirs, files in os.walk(skill_dir):
        for fn in files:
            if fn.endswith((".md", ".json")):
                rel = os.path.relpath(os.path.join(base, fn), skill_dir)
                raw = open(os.path.join(base, fn), encoding="utf-8").read()
                for block in re.split(r"\n(?:#{1,3} |-{3,}|\Z)", raw):
                    block = block.strip()
                    if len(block) > 40:
                        skill_corpus.append((rel, block))

    # open-book corpus: same blocking over admitted repo evidence files
    repo_corpus: list[tuple[str, str]] = []
    for rel in profile.files:
        p = os.path.join(repo_root, rel)
        try:
            raw = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for i in range(0, len(raw.splitlines()), 12):
            block = "\n".join(raw.splitlines()[i:i + 12])
            if len(block.strip()) > 40:
                repo_corpus.append((rel, block))

    results, closed_pass, open_pass = [], 0, 0
    closed_present = open_present = 0
    for q in qs["questions"]:
        qtoks = _tokens(q["question"])
        # gold keywords: the operational fact tokens that must be recoverable
        gtoks = _tokens(q.get("gold_keywords") or q["gold_answer"])
        thr = qs.get("recall_threshold", 0.6)

        # 'present': gold recall over the WHOLE corpus (is the fact carried
        # at all?) — separates distillation gaps from retrieval gaps.
        c_all = _tokens(" ".join(t for _p, t in skill_corpus))
        o_all = _tokens(" ".join(t for _p, t in repo_corpus))
        c_present = len(gtoks & c_all) / max(1, len(gtoks)) >= thr
        o_present = len(gtoks & o_all) / max(1, len(gtoks)) >= thr

        # 'retrieved': gold recall in the top-k passages an agent would open
        c_top = _top_passages(qtoks, skill_corpus, top_k)
        c_text = " ".join(t for _p, t in c_top)
        c_recall = len(gtoks & _tokens(c_text)) / max(1, len(gtoks))
        o_top = _top_passages(qtoks, repo_corpus, top_k)
        o_text = " ".join(t for _p, t in o_top)
        o_recall = len(gtoks & _tokens(o_text)) / max(1, len(gtoks))
        c_ok, o_ok = c_recall >= thr, o_recall >= thr
        closed_present += c_present
        open_present += o_present
        closed_pass += c_ok
        open_pass += o_ok
        results.append({
            "id": q["id"], "question": q["question"],
            "gold_keywords": q.get("gold_keywords") or q["gold_answer"],
            "evidence_file": q.get("evidence_file", ""),
            "closed_book": {"passages": [p for p, _ in c_top], "present": c_present,
                            "gold_recall": round(c_recall, 3), "pass": c_ok},
            "open_book": {"passages": [p for p, _ in o_top], "present": o_present,
                          "gold_recall": round(o_recall, 3), "pass": o_ok},
        })

    n = len(qs["questions"])
    closed_frac = closed_pass / n if n else 0.0
    open_frac = open_pass / n if n else 0.0
    closed_p_frac = closed_present / n if n else 0.0
    open_p_frac = open_present / n if n else 0.0
    return {
        "n_questions": n,
        "closed_book_pass": closed_pass,
        "open_book_pass": open_pass,
        "closed_present": closed_present,
        "open_present": open_present,
        "closed_frac": round(closed_frac, 3),
        "open_frac": round(open_frac, 3),
        "carry_ratio": round(closed_frac / open_frac, 3) if open_frac else 0.0,
        "present_carry_ratio": round(closed_p_frac / open_p_frac, 3) if open_p_frac else 0.0,
        "per_question": results,
    }


def write_record(skill_dir: str, gates: list[dict], qa: dict) -> str:
    """The paper's construction record R: checks performed + unresolved gaps,
    written OUTSIDE the runtime graph (verify/ next to skills/)."""
    out_dir = os.path.dirname(os.path.dirname(skill_dir.rstrip("/")))  # <out>/skills/x -> <out>
    rec = os.path.join(out_dir, "record.json")
    gaps = [g["detail"] for g in gates if not g["pass"]]
    misses = [r["id"] for r in qa["per_question"] if not r["closed_book"]["pass"]]
    if misses:
        gaps.append(f"QA closed-book misses: {misses}")
    json.dump({"gates": gates, "qa": {k: v for k, v in qa.items() if k != "per_question"},
               "unresolved_gaps": gaps}, open(rec, "w"), indent=2)
    return rec
