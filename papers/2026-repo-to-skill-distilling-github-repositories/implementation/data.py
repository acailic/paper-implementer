"""data.py — source ingestion for the Repo-To-Skill re-implementation.

Paper: "Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills"
        (Chen et al., 2026, arXiv:2609.02749)

Implements the ingestion half of the paper's *scoping* and *grounding*
stages for a task-agnostic anchor (z = a GitHub repository):

  * shallow-clone (or reuse) a repository checkout          [anchor z]
  * fix the include/exclude EVIDENCE BOUNDARY (App. A.1)     [scoping]
  * parse README + packaging metadata + Python ASTs into a
    SourceProfile of EvidenceUnits with file:line provenance [grounding]

Deterministic, CPU-only, stdlib-only. The repository is treated as
UNTRUSTED DATA: files are only ever read as bytes/text and parsed with
`ast.parse` (no execution of repo code, no imports from the checkout).

Run-together counterpart: model.py (Distiller), train.py (pipeline),
verify.py (SkillVerifier).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field

# ---------------------------------------------------------------- evidence boundary

# Extensions/names admitted as evidence (paper A.1: source roots, docs,
# examples, tests, scripts, configuration).
INCLUDE_EXTS = {".py", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".sh"}
INCLUDE_NAMES = {"Makefile", "Dockerfile", "LICENSE"}
# Excluded (paper A.1: generated files, build outputs, vendored deps,
# environments, caches, large artifacts).
EXCLUDE_DIRS = {".git", "__pycache__", ".github", ".venv", "venv", "node_modules",
                "build", "dist", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints",
                "assets", "data", "checkpoints", "out", "logs"}
MAX_FILE_BYTES = 200_000          # skip pathologically large blobs
MAX_FILES = 400


def evidence_boundary(root: str) -> list[str]:
    """Walk `root` and return admitted evidence files (relative paths, sorted).

    This is the paper's include/exclude map, applied before anything is
    parsed: keep the graph 'tied to public, reusable package behavior
    rather than arbitrary checkout state'.
    """
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for fn in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            ext = os.path.splitext(fn)[1].lower()
            if fn in INCLUDE_NAMES or ext in INCLUDE_EXTS:
                full = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(full) > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                out.append(rel)
                if len(out) >= MAX_FILES:
                    sys.stderr.write(f"[boundary] file cap {MAX_FILES} hit; truncating\n")
                    return out
    return out


# ---------------------------------------------------------------- provenance-carrying evidence

@dataclass
class EvidenceUnit:
    """One grounded piece of evidence: a claim + where it came from."""
    kind: str            # readme_section | readme_snippet | install_cmd | class | function
                         # | method | entry_point | dependency | module_docstring | meta
    name: str            # human label
    text: str            # the excerpt itself (trimmed)
    file: str            # provenance: repo-relative path
    line: int = 0        # provenance: 1-based line (0 = n/a)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "text": self.text,
                "file": self.file, "line": self.line}


@dataclass
class SourceProfile:
    """Everything the distiller is allowed to know about the source."""
    repo_url: str
    commit: str                      # provenance stamp (paper: source commit recorded)
    repo_name: str
    files: list[str] = field(default_factory=list)          # admitted evidence files
    evidence: list[EvidenceUnit] = field(default_factory=list)
    readme_install_cmds: list[str] = field(default_factory=list)
    dependencies: set = field(default_factory=set)

    def evidence_json(self) -> list[dict]:
        return [e.as_dict() for e in self.evidence]


# ---------------------------------------------------------------- clone / anchor

def clone(url: str, work_dir: str) -> tuple[str, str]:
    """Shallow-clone `url` into a fresh work_dir and return (repo_root, commit).

    Per the environment's untrusted-data policy the checkout lives in its
    own new directory and is never executed or imported from.
    """
    repo_root = os.path.join(work_dir, "repo")
    if os.path.exists(repo_root) and os.path.isdir(repo_root):
        sys.stderr.write(f"[anchor] reusing existing checkout {repo_root}\n")
    else:
        os.makedirs(work_dir, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", url, repo_root],
                       check=True, capture_output=True)
    commit = subprocess.run(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    return repo_root, commit


# ---------------------------------------------------------------- README parsing

_CODE_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)
_HEADER = re.compile(r"^(#{1,4})\s+(.*)$", re.M)
_INSTALL = re.compile(r"^\s*(pip install|conda install|python -m pip|uv pip|git clone).*",
                      re.M | re.I)


def parse_readme(root: str, rel: str) -> list[EvidenceUnit]:
    """Split the README into sections + tagged code snippets (grounding).

    Snippets are classified by content heuristics: install / training /
    *usage* / other. Each unit keeps file provenance; line numbers are
    approximated by prefix matching against the raw file.
    """
    path = os.path.join(root, rel)
    text = open(path, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    units: list[EvidenceUnit] = []

    def line_of(snippet_first_line: str) -> int:
        for i, ln in enumerate(lines):
            if snippet_first_line and snippet_first_line in ln:
                return i + 1
        return 0

    # first paragraph = purpose statement (used for the skill description)
    first_para = ""
    for ln in lines:
        s = ln.strip()
        if not s and first_para:
            break
        if s and not s.startswith(("#", "!", "<", "[!", "---", "|")) and not s.startswith("```"):
            first_para += (" " if first_para else "") + s
        elif first_para:
            break
    if first_para:
        units.append(EvidenceUnit("meta", "purpose", first_para[:500], rel, 1))

    # sections
    spans = [(m.start(), m.group(2).strip()) for m in _HEADER.finditer(text)]
    for i, (pos, title) in enumerate(spans):
        end = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        body = text[pos:end].strip()
        if len(body) > 4000:
            body = body[:4000] + "\n…(truncated)"
        units.append(EvidenceUnit("readme_section", title, body, rel, line_of(title)))

    # code snippets, classified
    for m in _CODE_FENCE.finditer(text):
        lang, code = m.group(1).lower(), m.group(2).strip()
        if not code or len(code) > 3000:
            continue
        if lang in ("bash", "sh", "shell", "console") or code.startswith(("pip ", "conda ", "python ", "git ")):
            kind = "install_cmd" if re.search(r"(pip|conda)\s+install", code) else "readme_snippet"
            tag = "shell"
        elif "import" in code or "=" in code:
            kind, tag = "readme_snippet", "python"
            if re.search(r"\.train\(|Trainer\(", code):
                tag = "python-training"
            elif re.search(r"model\.|generate\(|sample\(", code):
                tag = "python-inference"
        else:
            kind, tag = "readme_snippet", lang or "text"
        first = code.splitlines()[0]
        units.append(EvidenceUnit(kind, f"snippet[{tag}]", code, rel, line_of(first)))
        if kind == "install_cmd":
            for cmd in code.splitlines():
                cmd = cmd.strip()
                if re.match(r"(pip|conda)\s+install", cmd):
                    units.append(EvidenceUnit("install_cmd", "cmd", cmd, rel, line_of(cmd)))
    return units


# ---------------------------------------------------------------- Python AST parsing

def _sig(node: ast.AST) -> str:
    try:
        return ast.unparse(node.args)
    except Exception:
        return "…"


def parse_python(root: str, rel: str) -> list[EvidenceUnit]:
    """AST-mine one Python file: public API surface with provenance.

    Mirrors the paper's live-inspection step ('import names, public
    signatures') done statically: docstrings establish intent, source
    confirms API claims. Constructors (__init__) are kept — operationally
    they are the most load-bearing signature in the surface.
    """
    path = os.path.join(root, rel)
    src = open(path, encoding="utf-8", errors="replace").read()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # untrusted sources may warn
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return []
    units: list[EvidenceUnit] = []

    doc = ast.get_docstring(tree)
    if doc:
        units.append(EvidenceUnit("module_docstring", rel, doc.strip()[:800], rel, 1))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            doc = (ast.get_docstring(node) or "").strip()
            units.append(EvidenceUnit("class", node.name,
                                      (doc or "(no docstring)")[:600], rel, node.lineno))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and (not sub.name.startswith("_") or sub.name == "__init__"):
                    mdoc = (ast.get_docstring(sub) or "").strip()
                    units.append(EvidenceUnit(
                        "method", f"{node.name}.{sub.name}({(_sig(sub))[:120]})",
                        (mdoc or "(no docstring)")[:300], rel, sub.lineno))
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name) and t.id.isupper():
                            units.append(EvidenceUnit("method", f"{node.name}.{t.id}",
                                                      f"class attribute (default config value)",
                                                      rel, sub.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            doc = (ast.get_docstring(node) or "").strip()
            units.append(EvidenceUnit("function", f"{node.name}({(_sig(node))[:120]})",
                                      (doc or "(no docstring)")[:400], rel, node.lineno))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    units.append(EvidenceUnit("function", t.id,
                                              "module-level constant", rel, node.lineno))
    if "__main__" in src:
        units.append(EvidenceUnit("meta", "runnable_module",
                                  "file has an if __name__ == '__main__' block (runnable as a script)",
                                  rel, 0))
    return units


# ---------------------------------------------------------------- packaging metadata

def parse_packaging(root: str, files: list[str]) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    for rel in files:
        base = os.path.basename(rel)
        if base in ("setup.py", "pyproject.toml", "setup.cfg"):
            text = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
            for m in re.finditer(r'(?:name|version)\s*[=:]\s*["\']([^"\']+)', text):
                units.append(EvidenceUnit("meta", f"{base}:{m.group(1)}", m.group(0), rel, 0))
            for m in re.finditer(r'console_scripts.*', text):
                units.append(EvidenceUnit("entry_point", "console_scripts", m.group(0)[:200], rel, 0))
            for m in re.finditer(r'install_requires\s*=\s*\[(.*?)\]', text, re.S):
                units.append(EvidenceUnit("dependency", "setup.py:install_requires",
                                          m.group(1)[:400], rel, 0))
    if os.path.exists(os.path.join(root, "requirements.txt")):
        reqs = open(os.path.join(root, "requirements.txt"), encoding="utf-8",
                    errors="replace").read().strip()
        if reqs:
            units.append(EvidenceUnit("dependency", "requirements.txt", reqs[:400], "requirements.txt", 0))
    return units


def collect_dependencies(profile_evidence: list[EvidenceUnit], root: str, files: list[str]) -> set:
    """Top-level third-party imports across the codebase (dependency signal)."""
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    deps: set[str] = set()
    local = {os.path.splitext(f)[0].replace("/", ".") for f in files if f.endswith(".py")}
    local |= {f.split("/")[0] for f in files if "/" in f}
    for rel in files:
        if not rel.endswith(".py"):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(open(os.path.join(root, rel), encoding="utf-8",
                                      errors="replace").read())
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    deps.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                deps.add(node.module.split(".")[0])
    return {d for d in deps
            if d not in stdlib and d not in local and d not in {"__future__"}}


# ---------------------------------------------------------------- profile assembly

def build_profile(repo_url: str, repo_root: str, commit: str) -> SourceProfile:
    name = os.path.basename(repo_url.rstrip("/")) or "repo"
    profile = SourceProfile(repo_url=repo_url, commit=commit, repo_name=name)
    profile.files = evidence_boundary(repo_root)

    for rel in profile.files:
        low = rel.lower()
        if low.startswith("readme") or low.startswith("docs/readme"):
            profile.evidence.extend(parse_readme(repo_root, rel))
        elif rel.endswith(".py"):
            profile.evidence.extend(parse_python(repo_root, rel))
    profile.evidence.extend(parse_packaging(repo_root, profile.files))
    profile.dependencies = collect_dependencies(profile.evidence, repo_root, profile.files)
    return profile
