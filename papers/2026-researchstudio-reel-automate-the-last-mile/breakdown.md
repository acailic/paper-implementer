# ResearchStudio-Reel: Automate the Last Mile of Research from Paper to Poster, Video, and Blog

**arXiv 2607.04438** | cs.CV | Repo paper rank 2 | Iter 97

---

## Problem & Motivation

Research dissemination — turning an accepted paper into a poster, talk video, and blog post — is still a manual last mile done by hand in the days after camera-ready. Prior automation treats each artifact in isolation, re-extracts the paper from scratch each time, ships one-way renders (PDF/PNG/MP4 the author cannot reopen in PowerPoint or Word), and gates quality on soft VLM-preference scores that plateau while load-bearing sections still read empty.

Three recurring gaps in prior work:

- **G1 — Isolated extraction.** Each artifact pipeline re-derives figure crops, captions, and metadata independently, so cross-artifact factual consistency is left to the user.
- **G2 — One-way renders.** Most prior artifacts ship as PDF/PNG/MP4 the author cannot reopen and tweak in PowerPoint or Word.
- **G3 — Soft quality gates.** Quality is judged by continuous VLM-as-judge aesthetic scores, so a layout scoring 7.8/10 is accepted even when a section reads empty.

**Core idea:** Treat the last mile as a composition of thin agent-readable skills that share one upstream extractor and wrap deterministic primitives in a measured-fill loop whose exits are hard pass/fail render gates.

---

## Key Insight / Contribution

1. **Five-skill composition** (Paper2Assets, Paper2Poster, Paper2Video, Paper2Blog, Paper2Reel) turning one paper PDF into a print-ready poster, synchronized talk video, and bilingual blog.
2. **Every deliverable ships editable** in native tools (PowerPoint for poster+video deck, Word for blog).
3. **Paper2Reel** — unified interactive HTML viewer binding poster ↔ video ↔ blog into one navigable surface.
4. **Best poster quality** on Paper2Poster benchmark: aesthetic 3.52 vs author ground-truth 2.94, wins 84–93% of papers.

---

## Method

### Pipeline Overview (Figure 2)

```
Paper (PDF/link) → Paper2Assets → {metadata.json, figures/, paper.txt, ...}
                                        ↓
                        ┌───────────────┼───────────────┐
                        ↓               ↓               ↓
                  Paper2Poster    Paper2Video     Paper2Blog
                  (poster.html    (video.pptx,    (en.docx,
                   poster.pptx,    video.mp4,      zh.docx)
                   poster.pdf)     video_no_sub.mp4)
                        ↓               ↓               ↓
                        └───────────────┼───────────────┘
                                        ↓
                                    Paper2Reel
                                    (reel.html + alignment.json)
```

### Skill 1: Paper2Assets (Shared Upstream Extractor)

Runs once, produces a shared bundle consumed by all generators:

**Outputs:**
- Full body text with page breaks
- Detected figure captions + per-figure manifest
- Cleaned figure images
- Metadata (title, authors, institutes, venue, paper/code links)
- Structured nine-section summary: Problem, Motivation, Contribution, Method, Dataset/Benchmark, Key Result, Ablation, Headline Numbers, Takeaway (each with essential entry + supplementary entry + spoken audio script)
- Institution logos (from Wikimedia Commons) + QR codes
- Inventory manifest (PDF checksum, section IDs, narration clip list)

**Figure cleanup chain:**
1. Deterministic prefix: strip chrome residue, baked-in caption strips, uniform white margins
2. Visual-AI step: tight bounding box proposal
3. Fresh-context sub-agent verifier: re-reads original against proposed crop
4. Commit only on clean pass; split when one raster packs two figures
5. Every mode idempotent; raw extract backed up before any crop

**Key property:** The bundle is the only interface between Paper2Assets and downstream skills — downstream skills never re-open the PDF.

### Skill 2: Paper2Poster (§3.2)

**Design requirements:**
- A1: Composition without template explosion (4 orthogonal axes)
- A2: Fill loop that converges (discrete steps, categorical verdicts)
- A3: Page too large to re-read (never pulls whole poster into context)
- A4: Figures fill their cards (height cap, hard floor)
- A5: Faithithful editable export (DOM→PPTX with native shapes)

**Composition axes (A1):**
1. Column layout: full / half / three-column
2. Visual style: 11 interchangeable themes (each a self-contained CSS file)
3. Title-band header: 5 arrangements of venue logo, institute logos, QR codes
4. Scan-to-Read block: 1-QR / 2-QR

**Staged fill loop (A2):**

Each section measured as `fullRatio = h_content / h_card` (via `getBoundingClientRect`).

| Verdict | fullRatio | Remedy |
|---------|-----------|--------|
| EMPTY | < 0.70 | Append supplementary paragraph or promote optional section |
| SPARSE | 0.70–0.90 | Polish up: pad prose, enlarge widget |
| **FULL** | **0.90–1.00** | **Target band — leave untouched** |
| SPILLAGE | 1.00–1.10 | Polish down: tighten prose |
| OVERFLOW | > 1.10 | Drop supplementary paragraph or optional section |

Figure fill gate: Method figure must paint ≥70% of its card on at least one axis.

Oscillation dampening: (1) moves sized by signed pixel delta, (2) loop refuses to re-apply move that already overshot on a section, (3) on-disk round counter trips circuit breaker.

Render-time whitespace expand: a single pass after loop convergence stretches underfilled cards toward 98% by growing whitespace between rows (figures never resized, column bottoms never moved).

**Editable PowerPoint bridge (A5):**
- Walks DOM node by node
- Reads geometry from `getBoundingClientRect`, appearance from `getComputedStyle`
- Converts CSS pixels to PowerPoint EMU units at fixed canvas scale
- Node classification:
  - `p/h1-h6/li/td` → editable text frame (inline `strong`/`em` survive as mixed-style runs)
  - `img` → replaceable picture (print resolution, `object-fit: contain`)
  - `svg` → rasterized in same box
  - MathJax equation (`data-tex` attribute) → native OMML equation
  - Decorative `div`/section card → rounded rectangle with fill/border/gradient/drop shadow
- CSS colors normalized to RGBA via one-pixel canvas; `hyphens: auto` → OOXML soft hyphens

**Quality gates (mandatory deliverables gate):**
- Every section lands in FULL band (90–98% of card)
- No figure paints <70% of its card on both axes
- Optional RRP gate: held-out reader model can still answer questions from poster alone
- Deterministic check: rendered PDF matches intended fixed-size canvas

**Output formats:** poster.html, poster.pdf, poster.png, poster.pptx — all released as one bundle.

### Skill 3: Paper2Video (§3.3)

Uses full ppt-master workflow for deck authoring.

**Design requirements:**
- B1: Duration planned before rendering (not by truncating final MP4)
- B2: Attention guidance (narration-aligned visual highlights)
- B3: Captions in two delivery contexts (burned-in + clean for interactive reel)
- B4: Video addressable after export (timeline.json sidecar)
- B5: Deterministic media failure checks

**Narration & duration planning (B1):**
- Convert shared section narration into video script with stable section IDs
- Planner estimates before TTS, requests semantic rewrites when too long/short
- After rendering: duration report compares measured MP4 length with plan
- Small residual errors → bounded speech-rate fix; large errors → narration rewriting

**Visual highlights (B2):**
- Script → visual-cue requirement file
- Semantic anchors attached to visible objects (figure panel, equation block, table row)
- Cue resolver combines script + word timings + slide geometry + anchors → `visual_cues.json`
- Normalized geometry survives different resolutions
- Default style: `spotlight_laser` (soft emphasis + focus point marker)

**Output:** video.pptx (editable deck), video.mp4 (with burned-in subtitles), video_no_subtitles.mp4 (clean), timeline.json (section→audio/subtitle/slide/cue alignment).

**Quality gates (check_video_package.py):**
- video.pptx + both MP4s exist and are playable
- Audio stream present in final video
- Non-empty subtitle sidecar; final MP4 ≠ raw MP4 (byte-identical check)
- Visual cues: coverage, normalized geometry, semantic target IDs, timing within audio segment, no word-sized highlight boxes
- PPTX + rendered frames: no blank frames, no text overflow, no text-image overlap, no cropped content
- Duration: TTS rate plan present, rejects unsafe rate changes

### Skill 4: Paper2Blog (§3.4)

**Design requirements:**
- C1: One evidence base for two languages
- C2: Register controlled during writing (WeChat-style ZH, neutral research-blog EN)
- C3: Article-level figure selection (3–7 evidence visuals, not all figures)
- C4: Word document with embedded images, stable fonts, fixed filenames
- C5: Word layout editor-ready (no near-blank pages, no thumbnail images, no orphan tails)

**Pipeline:**
1. Build shared evidence map from Paper2Assets bundle (hook, problem, method, claims, results, limitations, source links, figure roles)
2. Language-specific outlines + voice (separate drafting, not sentence-by-sentence translation)
3. Shared figure set selection for article evidence
4. DOCX assembly with fixed filenames, embedded media, language-specific captions
5. Layout inspection (DOCX internal structure + rendered page images in strict mode)

**Quality gates:**
- Both Word files exist, readable, contain enough text
- Images embedded (not linked), no placeholder text
- Font declarations: Latin body font + Chinese fallback
- Bilingual checks: same number of embedded images, same identity/order, same numeric claims, same technical terms
- Layout: no underfilled images, no images pushed to next page unnecessarily, no orphan tails

### Skill 5: Paper2Reel (§3.5)

**Interaction design:**
- Poster-first (most compact map of the paper)
- Hover → immediate feedback; double-click → section modal
- Section modal: video on left, blog on right, draggable splitter
- Video pane uses subtitle-free source; captions from sidecar with toggle
- Slide thumbnails below video → seek on click
- Blog pane: matching article block, language switch

**Content alignment:**
- Alignment sidecar: each canonical section ID → poster block, slide targets, video start/end times, subtitle tracks, slide thumbnails, blog blocks
- Handles incomplete inputs: inspects which upstream deliverables are missing, completes missing stages through full workflows
- Keeps original artifacts intact (copied into organized folders)

**Quality gates:**
- Static: viewer, alignment record, manifest, poster assets, media directories, slide frames, blog blocks, downloads all exist
- Rejects: stale tabbed-viewer markers, machine-local path leaks, backup files, missing resources
- Browser: serves bundle with range-capable local server, opens in headless browser, checks poster-first loading, hover, section modal, split-pane, subtitle toggle, slide seeking, downloads, keyboard shortcuts

---

## Training Details

N/A — this is a systems/agent composition paper, not a trained model. Uses Claude Code (claude-opus-4.8), Codex (gpt-5.5), Edge TTS, headless Chromium (Playwright), LibreOffice, ffmpeg, python-docx as deterministic primitives.

---

## Results

### Table 1: Poster Quality on Paper2Poster Benchmark (100 papers)

Mean of two held-out VLM judges (claude-opus-4.8 + gpt-5.5). 1–5 scale; PaperQuiz in %.

| System | Elem. | Engag. | Layout | Low | Logic | Cont. | **Aesth. Mean** | Detail§ | Underst. | **Info. Mean** | Quiz% | HTML | PPTX | Video | Blog |
|--------|-------|--------|--------|-----|-------|-------|-----------------|---------|----------|----------------|-------|------|------|-------|------|
| Claude-4.8 Opus | 2.97 | 2.98 | 2.98 | 3.95 | 4.00 | 3.90 | 3.41 | 3.67 | 3.71 | 3.80 | 55.43 | × | × | × | × |
| GPT-5.5 | 2.48 | 2.88 | 2.92 | 3.98 | 3.92 | 3.98 | 3.33 | 3.37 | 3.39 | 3.59 | 53.43 | × | × | × | × |
| Gemini-3.1 Pro | 2.83 | 3.22 | 3.17 | 3.95 | 3.98 | 3.95 | 3.55 | 3.80 | 4.00 | 3.90 | 57.45 | × | × | × | × |
| Paper2Poster Tool* | 1.82 | 2.68 | 2.93 | 2.62 | 2.92 | 2.87 | 2.64 | 3.76 | 3.70 | 3.68 | 75.40 | × | × | × | × |
| P2P* | 1.20 | 2.34 | 2.23 | 1.52 | 3.15 | 3.31 | 2.29 | 3.92 | 3.78 | 3.85 | 95.65 | × | × | × | × |
| PosterGen* | 3.14 | 3.33 | 3.41 | 3.82 | 3.97 | 4.00 | 3.78 | 3.39 | 3.71 | 3.80 | 63.00 | × | ✓ | × | ✓ |
| **RS-Reel (Codex)** | — | — | — | — | — | — | **3.36** | — | — | — | — | ✓ | ✓ | ✓ | ✓ |
| **RS-Reel (Claude Code)** | — | — | — | — | — | — | **3.52** | — | — | — | — | ✓ | ✓ | ✓ | ✓ |
| Author ground-truth | 3.02 | 2.49 | 3.31 | 3.80 | 3.68 | 3.41 | **3.28** | 3.63 | 3.92 | **3.78** | 50.11 | — | — | — | — |

Note: Codex row is on a benchmark subset (full set pending). Individual sub-criterion cells for Codex not shown in Table 1; per-judge breakdown in Table 5.

**Key results:**
- RS-Reel (Claude Code) aesthetic mean 3.52 > author ground-truth 2.94 (margin +0.58)
- RS-Reel (Claude Code) information mean 3.90 > author 3.63 (margin +0.27)
- Wins overall score on 84–93% of papers
- Quiz ordering inversely correlates with aesthetic ordering (density vs legibility tension)
- Author posters last on both Quiz splits (human designer prunes to headline results)

**Ablation:**
- Same model (claude-opus-4.8), single-shot → aesthetic 2.76, Layout 2.83; with fill loop → aesthetic 3.52, Layout 3.97 (gain +0.76 aesthetic, +1.14 Layout)
- Same model (gpt-5.5), single-shot → aesthetic 3.03, Layout 3.22; with fill loop → aesthetic 3.36, Layout 3.82 (gain +0.33 aesthetic, +0.60 Layout)
- Skill machinery helps regardless of base model

### Table 2: Paper2Video Capability Audit

| System | Highlight | Duration | PPTX | Subtitles | Audio |
|--------|-----------|----------|------|-----------|-------|
| Deck Tools (Gamma, Canva) | × | × | × | × | ✓ |
| Video Tools (Synthesia, HeyGen) | × | × | × | × | ✓ |
| NotebookLM Video | ✓ | × | × | × | ✓ |
| Paper-to-video Agents | × | ✓ | ✓ | × | ✓ |
| **ResearchStudio-Reel** | **✓** | **✓** | **✓** | **✓** | **✓** |

### Table 3: Paper2Blog Capability Audit

| System | Layout | Figures | DOCX | Bilingual | Summary |
|--------|--------|---------|------|-----------|---------|
| Semantic Scholar TLDR | × | × | × | × | ✓ |
| Research Assistants (Elicit, SciSpace) | × | × | × | × | ✓ |
| Scholarcy | × | × | ✓ | × | ✓ |
| NotebookLM | × | × | ✓ | × | ✓ |
| **ResearchStudio-Reel** | **✓** | **✓** | **✓** | **✓** | **✓** |

### Table 4: Per-Stage Cost Breakdown (mean over 5 papers, claude-opus-4.8)

Full pipeline: **~89.2 min, ~2,568K total input tokens, ~276K output tokens**

| Skill | Stage | Time (min) | Turns | Input (K) | Share% | Cache (K) | Output (K) | Share% |
|-------|-------|------------|-------|-----------|--------|-----------|------------|--------|
| Paper2Assets | extract & figures | 8.6 | 112 | 461 | 18% | 12,068 | 33 | 12% |
| Paper2Poster | compose | 5.1 | 43 | 166 | 7% | 7,989 | 12 | 4% |
| | **fill loop** | **14.8** | **100** | **310** | **12%** | **25,185** | **48** | **18%** |
| | render | 1.9 | 19 | 18 | <1% | 5,527 | 6 | 2% |
| | narration audio | 1.5 | 3 | 5 | <1% | 916 | 1 | <1% |
| | subtotal | 23.3 | 166 | 523 | 20% | 39,616 | 67 | 24% |
| Paper2Video | script & cue spec | 4.7 | 49 | 512 | 11% | 5,385 | 20 | 7% |
| | deck (ppt-master) | 5.7 | 38 | 90 | 4% | 6,665 | 16 | 6% |
| | visual cues | 1.3 | 16 | 21 | <1% | 2,925 | 6 | 2% |
| | narration audio | 3.4 | 21 | 33 | 1% | 2,857 | 7 | 3% |
| | render & mux | 9.2 | 38 | 43 | 2% | 7,149 | 17 | 6% |
| | QA gate | 4.0 | 30 | 44 | 2% | 6,315 | 14 | 5% |
| | subtotal | 28.5 | 193 | 744 | 20% | 31,297 | 80 | 29% |
| Paper2Blog | figures & setup | 6.6 | 47 | 524 | 20% | 4,849 | 17 | 6% |
| | DOCX assembly | 0.1 | 2 | 4 | <1% | 336 | 1 | <1% |
| | QA gate & revision | 10.0 | 63 | 243 | 9% | 11,062 | 47 | 17% |
| | subtotal | 16.7 | 112 | 771 | 30% | 16,247 | 64 | 23% |
| Paper2Reel | plan | 3.2 | 24 | 198 | 8% | 1,910 | 7 | 2% |
| | assemble | 2.7 | 16 | 26 | <1% | 1,359 | 4 | 1% |
| | QA gate | 6.2 | 52 | 77 | 3% | 6,049 | 21 | 8% |
| | subtotal | 12.1 | 92 | 301 | 12% | 9,318 | 32 | 12% |
| **Full pipeline** | | **89.2** | **675** | **2,568** | **100%** | **108,546** | **276** | **100%** |

Heaviest stages: fill loop (12% input, 18% output), extract & figures (18% input, 12% output), deck ppt-master (4% input, 6% output).

Most traffic is cached-context re-reads (billed at ~10% of fresh input), so dollar cost tracks the much smaller Input and Output columns.

### Table 5: Per-Judge Benchmark Scores (Appendix D)

**Claude judge (claude-opus-4.8):**

| System | El. | En. | La. | Lo. | Lg. | Co. | Det.% | Und.% |
|--------|-----|-----|-----|-----|-----|-----|-------|-------|
| Claude-4.8 Opus | 2.97 | 2.57 | 2.77 | 4.00 | 4.00 | 3.97 | 69.33 | 96.47 |
| GPT-5.5 | 2.97 | 2.83 | 3.20 | 4.00 | 4.00 | 3.97 | 68.87 | 96.60 |
| Gemini-3.1 Pro | 3.00 | 2.90 | 3.00 | 3.97 | 3.97 | 3.93 | 64.60 | 95.47 |
| Paper2Poster Tool* | 2.00 | 1.25 | 1.38 | 2.50 | 3.00 | 3.14 | 77.73 | 97.93 |
| P2P* | 2.73 | 2.47 | 3.23 | 3.97 | 3.83 | 3.70 | 62.15 | 97.11 |
| PosterGen* | 2.93 | 2.26 | 2.96 | 3.96 | 3.82 | 3.63 | 73.13 | 95.79 |
| RS-Reel (Claude Code) | 3.29 | 3.04 | 3.46 | 3.97 | 3.86 | 3.94 | 60.43 | 95.80 |
| RS-Reel (Codex) | — | — | — | — | — | — | — | — |
| Author GT | 3.14 | 2.51 | 3.16 | 3.59 | 3.63 | 3.23 | 53.24 | 95.06 |

**GPT judge (gpt-5.5):**

| System | El. | En. | La. | Lo. | Lg. | Co. | Det.% | Und.% |
|--------|-----|-----|-----|-----|-----|-----|-------|-------|
| Claude-4.8 Opus | 2.97 | 2.40 | 2.90 | 3.90 | 4.00 | 3.83 | 64.00 | 90.27 |
| GPT-5.5 | 3.00 | 2.93 | 3.23 | 3.97 | 3.97 | 3.87 | 62.40 | 89.80 |
| Gemini-3.1 Pro | 2.97 | 2.93 | 3.33 | 3.93 | 3.93 | 3.40 | 58.27 | 89.40 |
| Paper2Poster Tool* | 1.62 | 1.38 | 1.50 | 2.25 | 3.00 | 2.89 | 73.13 | 93.27 |
| P2P* | 2.63 | 2.20 | 2.19 | 2.99 | 3.04 | 2.47 | 52.81 | 86.86 |
| PosterGen* | 2.93 | 3.07 | 3.00 | 3.98 | 3.82 | 3.87 | 50.43 | 85.96 |
| RS-Reel (Claude Code) | 3.36 | 3.25 | 3.59 | 4.02 | 4.00 | 3.97 | 50.43 | 85.96 |
| RS-Reel (Codex) | — | — | — | — | — | — | — | — |
| Author GT | 2.89 | 2.47 | 3.16 | 3.23 | 3.18 | 3.50 | 46.98 | 82.78 |

---

## Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────┐
│                   Paper (PDF / arXiv link)           │
└──────────────────────┬──────────────────────────────┘
                       │ /paper2assets "Convert my paper to asset"
                       ▼
              ┌─────────────────────┐
              │    Paper2Assets     │  Shared Extractor
              │  ┌───────────────┐  │
              │  │ Text + captions│  │  Figures (cleaned)
              │  │ 9-sec summary │  │  Logos + QR codes
              │  │ Metadata      │  │  Inventory manifest
              │  │ Narration list│  │
              │  └───────────────┘  │
              └────────┬────────────┘
                       │ shared bundle (read verbatim)
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │P2Poster  │ │P2Video   │ │P2Blog    │
   │          │ │          │ │          │
   │ Compose  │ │ Script   │ │ Evidence │
   │ Fill Loop│ │ Deck     │ │   map    │
   │ (5-bands)│ │ (ppt-msr)│ │ ZH+EN    │
   │ Render   │ │ Audio    │ │   draft  │
   │ Gate     │ │ Cues     │ │ DOCX asm │
   │ →HTML/PDF│ │ Render   │ │ Gate     │
   │ →PPTX    │ │ →MP4×2   │ │ →en.docx │
   └────┬─────┘ │ Gate     │ │ →zh.docx │
        │       └────┬─────┘ └────┬─────┘
        │            │            │
        └────────────┼────────────┘
                     │ /paper2reel Create ...
                     ▼
              ┌─────────────────────┐
              │    Paper2Reel        │  Convergence Layer
              │  ┌───────────────┐  │
              │  │ reel.html     │  │  Interactive viewer
              │  │ alignment.json│  │  Section-level nav
              │  │ manifest.json│  │  Downloads
              │  └───────────────┘  │
              └─────────────────────┘
```

---

## Limitations

1. **Figure-cleanup residue (L1):** Caption strip or body-text slice baked into raster survives deterministic prefix — visual-AI verifier re-crops.
2. **Fill-loop non-convergence (L2):** Discrete loop can oscillate across 90–98% band — circuit breaker ships best-measured state.
3. **Slide–narration referential drift (L3):** Script says "Figure 3" while slide shows Figure 2 — QA gate catches via alignment timeline.
4. **Bilingual blog drift (L4):** Two language drafts disagree on numeric result — evidence map + DOCX gate cross-checks.
5. **Voice mismatched narration (L5):** Default Edge TTS voice reads keynote-style deck flat — re-run with different voice.
6. **Domain scope:** Calibrated on ML/CV/NLP venues; untested on biomedicine, physics, design-heavy fields.
7. **Evaluation coverage:** Quantitative evaluation v1 is graded-benchmark only on poster side; video and blog compared on capability coverage only.
8. **No bespoke figure synthesis:** Pipeline reuses only figures from the paper; cannot draw custom method/overview diagrams like a human designer would.

---

## Honest Scope Issues

1. **No human evaluation.** Only VLM-as-judge + capability audits. No controlled human reading-and-recall study across any artifact.
2. **Codex row on benchmark subset.** Table 1 Codex cells scored on subset, not full 100-paper benchmark. "Ordering already stable" but not numerically reported.
3. **Evaluation is proxy-bound.** PaperQuiz (text density) and aesthetic rubric pull in opposite directions — neither measures actual reader understanding.
4. **Figure sources only.** Cannot generate bespoke method diagrams, icons, or explanatory visuals. Gap to human posters is generative, not compositional.
5. **Aesthetic rubric gameable.** Fill loop optimizes geometric density target, not semantic correctness. A denser poster always wins PaperQuiz; a sparser one wins aesthetics.
6. **Claude Code + Codex dependency.** Paper explicitly states skill machinery runs on Claude Code and Codex runtimes. No claim of model-agnostic portability beyond those two.
7. **Single benchmark.** Only Paper2Poster (100 papers). No video or blog benchmarks exist, but capability-audit-only comparison is inherently weaker.
8. **Cost not fully reported.** Table 4 reports tokens and time, but dollar cost not stated explicitly. Cache billing (~10%) mentioned qualitatively.
9. **No statistical significance.** Two VLM judges, no confidence intervals, no multiple seeds, no hypothesis testing on any comparison.
10. **Author ground-truth is heterogeneous.** Different authors have different poster styles; aggregate comparison may hide per-paper quality variance.
