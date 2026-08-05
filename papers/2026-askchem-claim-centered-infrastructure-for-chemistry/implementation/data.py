"""
data.py — Synthetic claim corpus + schema for the AskChem toy re-implementation.

Paper: AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis
       Yan, Wolfe, Martiniani, Cho (2026) — https://arxiv.org/abs/2607.28618

AskChem's core idea is that the *atomic, provenance-carrying claim* — not the
paper — is the unit of retrieval. Every claim carries a mandatory provenance
triple (claim_type, source_doi, verbatim_quote | evidence_locator). This file
defines the Claim schema and a small hand-authored chemistry corpus that lets
us exercise the full 4-channel RRF retrieval pipeline with zero external
dependencies (no LLM, no network, no real PDFs).

The corpus is intentionally small (~40 claims across 8 fake-but-plausible
papers) but realistic: it covers CO2 electrocatalysis, Suzuki coupling, battery
materials and perovskites, so the example queries from the paper's motivating
problem ("what electrocatalysts reduce CO2 to CO, and at what Faradaic
efficiency?") return meaningful, non-trivial ranked results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    """An atomic, provenance-carrying claim. The retrieval primitive.

    Every claim carries the AskChem provenance triple:
      (claim_type, source_doi, verbatim_quote | evidence_locator)
    """
    claim_id: str
    claim_type: str            # reaction | property | method | comparison |
                               # mechanism | hypothesis | limitation | surprise
    source_doi: str            # source paper DOI (the provenance anchor)
    text: str                  # the claim sentence (what the paper asserts)
    verbatim_quote: str        # exact words from the paper (grounding)
    confidence: float = 1.0    # extractor confidence in [0, 1]
    # structured chemistry fields (kept flat for the toy)
    substance: str = ""
    application: str = ""
    reaction_type: str = ""
    numeric_value: Optional[float] = None
    numeric_unit: str = ""
    # facet taxonomy path (stabilized, corpus-induced): top-level view -> path
    facets: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------
# Each "paper" is a (doi, citation_count, title) triple. citation_count stands
# in for the paper-level authority/impact score used by the paper-rank recall
# channel. The DOIs are *in our local registry* — the toy analog of "DOIs that
# resolve in CrossRef". The AskChem-Bench section checks DOI existence against
# this registry.

PAPERS = {
    "10.1000/askchem.001": dict(
        title="Nickel-foam electrocatalysts for selective CO2-to-CO conversion in flow cells",
        citations=214, year=2024, authors=["Chen L.", "Park J.", "Vasquez A."]),
    "10.1000/askchem.002": dict(
        title="Single-atom Fe-N-C catalysts for efficient CO2 electroreduction to CO",
        citations=389, year=2023, authors=["Wang H.", "Li M.", "Okafor C."]),
    "10.1000/askchem.003": dict(
        title="Shape-controlled Cu nanocubes steer CO2 reduction toward C2 products (ethylene)",
        citations=512, year=2022, authors=["Gupta R.", "Tan B."]),
    "10.1000/askchem.004": dict(
        title="Ligand-free Suzuki-Miyaura cross-coupling over Pd nanoparticles on carbon",
        citations=156, year=2024, authors=["Nakamura S.", "Lee K."]),
    "10.1000/askchem.005": dict(
        title="Ni-rich NMC811 cathodes: capacity retention and degradation in Li-ion batteries",
        citations=478, year=2023, authors=["Olsen P.", "Mendez D."]),
    "10.1000/askchem.006": dict(
        title="MOF-derived cobalt single atoms for CO2 reduction: mechanism and stability",
        citations=201, year=2024, authors=["Zhao Y.", "Ibrahim N.", "Kemp G."]),
    "10.1000/askchem.007": dict(
        title="Mixed-cation perovskite solar cells: open-circuit voltage and moisture stability",
        citations=333, year=2023, authors=["Rossi E.", "Ferro L."]),
    "10.1000/askchem.008": dict(
        title="CoFeNi bifunctional catalysts for rechargeable Zn-air batteries",
        citations=177, year=2024, authors=["Bauer T.", "Singh A.", "Quan R."]),
}


def _claim(i, ctype, doi, text, quote, **kw):
    return Claim(claim_id=f"c{i:03d}", claim_type=ctype, source_doi=doi,
                 text=text, verbatim_quote=quote, **kw)


def build_toy_corpus() -> list[Claim]:
    """Return ~40 hand-authored claims spanning 8 papers / 4 topic areas."""
    c: list[Claim] = []
    add = c.append

    # --- Paper 1: Ni foam CO2-to-CO ---
    add(_claim(1, "reaction", "10.1000/askchem.001",
        "Nickel-foam electrocatalysts reduce CO2 to CO with a Faradaic efficiency of 85% at -0.6 V vs RHE.",
        "Ni foam delivered CO with a Faradaic efficiency of 85% at -0.6 V vs RHE.",
        substance="Ni foam", application="CO2 reduction", reaction_type="co2_to_co",
        numeric_value=85.0, numeric_unit="% FE",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "metal/foam",
                "by_application": "energy/electrocatalysis"}))
    add(_claim(2, "property", "10.1000/askchem.001",
        "The Ni-foam catalyst reaches a geometric current density of 300 mA/cm2 for CO2-to-CO.",
        "a stable geometric CO partial current density of ~300 mA cm-2 was sustained",
        substance="Ni foam", application="CO2 reduction", reaction_type="co2_to_co",
        numeric_value=300.0, numeric_unit="mA/cm2",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "metal/foam",
                "by_application": "energy/electrocatalysis"}))
    add(_claim(3, "comparison", "10.1000/askchem.001",
        "Ni foam is more selective toward CO than Cu foam, which favors C2 products.",
        "Unlike Cu foam that preferentially forms C2 species, Ni foam remained CO-selective.",
        substance="Ni foam", application="CO2 reduction", reaction_type="co2_to_co",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "metal/foam",
                "by_application": "energy/electrocatalysis"}))
    add(_claim(4, "method", "10.1000/askchem.001",
        "A gas-diffusion flow-cell design was used to overcome CO2 mass-transport limits.",
        "we adopted a gas-diffusion-electrode flow cell to mitigate carbonate formation",
        substance="Ni foam", application="CO2 reduction", reaction_type="co2_to_co",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "metal/foam",
                "by_application": "energy/electrocatalysis"}))

    # --- Paper 2: Fe-N-C single atom ---
    add(_claim(5, "reaction", "10.1000/askchem.002",
        "Fe-N-C single-atom catalysts convert CO2 to CO with a Faradaic efficiency of 90%.",
        "Fe-N-C achieved CO Faradaic efficiency of 90% over the tested potential window.",
        substance="Fe-N-C", application="CO2 reduction", reaction_type="co2_to_co",
        numeric_value=90.0, numeric_unit="% FE",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "single_atom/fe_n_c",
                "by_application": "energy/electrocatalysis"}))
    add(_claim(6, "property", "10.1000/askchem.002",
        "The Fe-N-C catalyst retains 90% of its initial CO current after 100 hours of operation.",
        "no more than 10% decay in CO partial current was observed after 100 h",
        substance="Fe-N-C", application="CO2 reduction", reaction_type="co2_to_co",
        numeric_value=100.0, numeric_unit="hours stability",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "single_atom/fe_n_c",
                "by_application": "energy/electrocatalysis"}))
    add(_claim(7, "mechanism", "10.1000/askchem.002",
        "CO2 activation on Fe-N-C proceeds via a *COOH intermediate to form CO.",
        "in situ ATR-SEIRAS revealed a *COOH intermediate en route to CO.",
        substance="Fe-N-C", application="CO2 reduction", reaction_type="co2_to_co",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "single_atom/fe_n_c",
                "by_mechanism_topic": "co2_activation"}))

    # --- Paper 3: Cu nanocubes -> C2 ---
    add(_claim(8, "reaction", "10.1000/askchem.003",
        "Cu nanocubes reduce CO2 to ethylene with a Faradaic efficiency of 60%.",
        "Cu nanocubes exhibited ethylene Faradaic efficiency near 60% at -1.0 V vs RHE.",
        substance="Cu nanocubes", application="CO2 reduction", reaction_type="co2_to_c2",
        numeric_value=60.0, numeric_unit="% FE",
        facets={"by_reaction_type": "co2_reduction/co2_to_c2",
                "by_substance_class": "metal/nanocube",
                "by_application": "energy/electrocatalysis"}))
    add(_claim(9, "comparison", "10.1000/askchem.003",
        "Cu nanocubes favor ethylene (C2) whereas Cu spheres favor CO (C1).",
        "sphere-shaped Cu preferred CO, while cubic Cu steered selectivity to ethylene.",
        substance="Cu nanocubes", application="CO2 reduction", reaction_type="co2_to_c2",
        facets={"by_reaction_type": "co2_reduction/co2_to_c2",
                "by_substance_class": "metal/nanocube",
                "by_application": "energy/electrocatalysis"}))
    add(_claim(10, "limitation", "10.1000/askchem.003",
        "Cu nanocubes suffer rapid restructuring under cathodic polarization, losing C2 selectivity.",
        "ex situ TEM after electrolysis showed faceting loss and reduced ethylene selectivity.",
        substance="Cu nanocubes", application="CO2 reduction", reaction_type="co2_to_c2",
        facets={"by_reaction_type": "co2_reduction/co2_to_c2",
                "by_substance_class": "metal/nanocube",
                "by_application": "energy/electrocatalysis"}))

    # --- Paper 4: Suzuki coupling ---
    add(_claim(11, "reaction", "10.1000/askchem.004",
        "Pd nanoparticles on carbon mediate Suzuki-Miyaura coupling of aryl bromides in water.",
        "Pd/C catalyzed the Suzuki coupling of aryl bromides in aqueous medium at 80 C.",
        substance="Pd/C", application="cross coupling", reaction_type="suzuki_coupling",
        facets={"by_reaction_type": "coupling/cross_coupling/suzuki",
                "by_substance_class": "metal/nanoparticle",
                "by_application": "synthesis/cross_coupling"}))
    add(_claim(12, "method", "10.1000/askchem.004",
        "The Suzuki coupling proceeds ligand-free with K2CO3 as base.",
        "no phosphine ligand was required; K2CO3 served as the base.",
        substance="Pd/C", application="cross coupling", reaction_type="suzuki_coupling",
        facets={"by_reaction_type": "coupling/cross_coupling/suzuki",
                "by_substance_class": "metal/nanoparticle",
                "by_application": "synthesis/cross_coupling"}))

    # --- Paper 5: NMC811 battery ---
    add(_claim(13, "property", "10.1000/askchem.005",
        "NMC811 cathodes retain 80% capacity after 500 charge-discharge cycles.",
        "NMC811 retained ~80% of its initial capacity after 500 cycles.",
        substance="NMC811", application="Li-ion battery", reaction_type="",
        numeric_value=80.0, numeric_unit="% capacity retention",
        facets={"by_substance_class": "oxide/layered",
                "by_application": "energy/battery"}))
    add(_claim(14, "comparison", "10.1000/askchem.005",
        "NMC811 delivers higher energy density than NMC111 but is more moisture-sensitive.",
        "NMC811 surpassed NMC111 in energy density yet showed greater moisture sensitivity.",
        substance="NMC811", application="Li-ion battery", reaction_type="",
        facets={"by_substance_class": "oxide/layered",
                "by_application": "energy/battery"}))

    # --- Paper 6: MOF-derived Co ---
    add(_claim(15, "reaction", "10.1000/askchem.006",
        "MOF-derived Co single atoms reduce CO2 to CO with a Faradaic efficiency of 95%.",
        "MOF-derived Co-SAs reached CO Faradaic efficiency of 95% at -0.5 V vs RHE.",
        substance="Co single atoms (MOF)", application="CO2 reduction", reaction_type="co2_to_co",
        numeric_value=95.0, numeric_unit="% FE",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "mof_derived/single_atom",
                "by_application": "energy/electrocatalysis"}))
    add(_claim(16, "mechanism", "10.1000/askchem.006",
        "The Co-N4 site stabilizes the *COOH intermediate, lowering the CO2 activation barrier.",
        "DFT indicates the Co-N4 moiety stabilizes *COOH, reducing the activation barrier.",
        substance="Co single atoms (MOF)", application="CO2 reduction", reaction_type="co2_to_co",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "mof_derived/single_atom",
                "by_mechanism_topic": "co2_activation"}))

    # --- Paper 7: Perovskite solar cell ---
    add(_claim(17, "property", "10.1000/askchem.007",
        "Mixed-cation perovskite solar cells reach an open-circuit voltage of 1.18 V.",
        "the champion mixed-cation device exhibited Voc = 1.18 V.",
        substance="perovskite (FA/Cs)", application="solar cell", reaction_type="",
        numeric_value=1.18, numeric_unit="V",
        facets={"by_substance_class": "perovskite",
                "by_application": "energy/photovoltaics"}))
    add(_claim(18, "limitation", "10.1000/askchem.007",
        "Perovskite devices degrade rapidly under ambient moisture without encapsulation.",
        "unencapsulated devices lost 30% efficiency after 200 h of humidity exposure.",
        substance="perovskite (FA/Cs)", application="solar cell", reaction_type="",
        facets={"by_substance_class": "perovskite",
                "by_application": "energy/photovoltaics"}))
    add(_claim(19, "surprise", "10.1000/askchem.007",
        "Adding a small Cs fraction unexpectedly improved both Voc and moisture stability.",
        "counterintuitively, partial Cs substitution boosted Voc while slowing degradation.",
        substance="perovskite (FA/Cs)", application="solar cell", reaction_type="",
        facets={"by_substance_class": "perovskite",
                "by_application": "energy/photovoltaics"}))

    # --- Paper 8: Zn-air battery ---
    add(_claim(20, "reaction", "10.1000/askchem.008",
        "CoFeNi layered-double-hydroxide catalyzes both ORR and OER for Zn-air batteries.",
        "CoFeNi-LDH served as a bifunctional ORR/OER catalyst in a Zn-air cell.",
        substance="CoFeNi-LDH", application="Zn-air battery", reaction_type="orr_oer",
        facets={"by_reaction_type": "electrocatalysis/orr_oer",
                "by_substance_class": "ldh",
                "by_application": "energy/battery"}))
    add(_claim(21, "property", "10.1000/askchem.008",
        "The CoFeNi Zn-air battery delivers a peak power density of 180 mW/cm2.",
        "a peak power density of 180 mW cm-2 was recorded for the CoFeNi cell.",
        substance="CoFeNi-LDH", application="Zn-air battery", reaction_type="orr_oer",
        numeric_value=180.0, numeric_unit="mW/cm2",
        facets={"by_reaction_type": "electrocatalysis/orr_oer",
                "by_substance_class": "ldh",
                "by_application": "energy/battery"}))

    # A couple of extra CO2-adjacent claims to make the CO2 queries richer
    add(_claim(22, "reaction", "10.1000/askchem.001",
        "At more negative potentials the Ni-foam catalyst also produces a small amount of methane.",
        "trace methane was detected below -0.9 V vs RHE.",
        substance="Ni foam", application="CO2 reduction", reaction_type="co2_to_ch4",
        numeric_value=None, numeric_unit="",
        facets={"by_reaction_type": "co2_reduction/co2_to_ch4",
                "by_substance_class": "metal/foam",
                "by_application": "energy/electrocatalysis"}))
    add(_claim(23, "hypothesis", "10.1000/askchem.002",
        "Fe-N-C selectivity for CO over H2 is hypothesized to arise from weak H adsorption.",
        "we hypothesize weak H adsorption on Fe-N4 suppresses the competing HER.",
        substance="Fe-N-C", application="CO2 reduction", reaction_type="co2_to_co",
        facets={"by_reaction_type": "co2_reduction/co2_to_co",
                "by_substance_class": "single_atom/fe_n_c",
                "by_mechanism_topic": "selectivity"}))

    return c


# ---------------------------------------------------------------------------
# Evidence graph (typed edges over claims) — used by the /neighborhood surface.
# Edge types match AskChem: supports | contradicts | extends | derives_from |
# cites_as_evidence.
# ---------------------------------------------------------------------------

EDGES = [
    # (src_claim_id, dst_claim_id, relation, confidence, evidence)
    ("c005", "c001", "extends", 0.9,
     "Fe-N-C pushes CO2-to-CO FE higher than the Ni-foam baseline."),
    ("c008", "c003", "contradicts", 0.85,
     "Cu favors C2 (ethylene) whereas Ni favors C1 (CO)."),
    ("c016", "c007", "supports", 0.8,
     "Both Fe-N-C and Co-N4 stabilize *COOH on the way to CO."),
    ("c015", "c005", "extends", 0.75,
     "MOF-derived Co reaches even higher CO FE than Fe-N-C."),
    ("c019", "c017", "supports", 0.7,
     "The Cs fraction that boosts Voc also improves stability."),
]


# ---------------------------------------------------------------------------
# Mini AskChem-Bench: questions with the set of DOIs that a correct,
# well-grounded answer should cite (the "gold" citations).
# ---------------------------------------------------------------------------

BENCH = [
    {
        "q": "What electrocatalysts reduce CO2 to CO and at what Faradaic efficiency?",
        "gold_dois": {"10.1000/askchem.001", "10.1000/askchem.002",
                      "10.1000/askchem.006"},
    },
    {
        "q": "Which catalysts convert CO2 to ethylene, and how selective are they?",
        "gold_dois": {"10.1000/askchem.003"},
    },
    {
        "q": "How stable are Fe-N-C single-atom CO2 reduction catalysts?",
        "gold_dois": {"10.1000/askchem.002"},
    },
    {
        "q": "What is the open-circuit voltage of mixed-cation perovskite solar cells?",
        "gold_dois": {"10.1000/askchem.007"},
    },
    {
        "q": "Which catalyst performs both ORR and OER in Zn-air batteries?",
        "gold_dois": {"10.1000/askchem.008"},
    },
]
