# Notes — The Verification Horizon: No Silver Bullet for Coding Agent Rewards

> Prvi i drugi prolaz kroz papir. Sirove misli, što sam primijetio.

## Šta je ovo za papir?

Praktični rad iz Qwen tima. Nema novog modela ni novog benchmarka — sistematski pregledaju kako projektovati reward signale za trening kodirajućih agenata. Glavna teza: nijedan fiksni reward ne može ostati efikasan dok se politika modela poboljšava; verifikacija mora ko-evoluirati sa generatorom.

| # | Tema | Sekcija | Reward tip |
|---|------|---------|------------|
| 1 | SWE zadaci — test-driven rewards | §2 | Izvršni testovi (pass/fail) |
| 2 | Frontend zadaci — interaktivni sudac | §3 | Rubrika + agentic interactive judge |
| 3 | Pravi svijet — korisnički feedback | §4 | Ljudski implicitni reward signali |
| 4 | Dugohorizontni zadaci — agent evaluator | §5 | Automatski agentic evaluator |

## Tri dimenzije kvaliteta verifikacije

1. **Scalability** — može li se signal proizvesti jeftino u skali?
2. **Faithfulness** — koliko odražava stvarnu korisničku namjeru?
3. **Robustness** — odolijeva li diverzifikovanim i adversarial inputima, te optimizacionom pritisku generatora?

Nijedan postojeći pristup ne zadovoljava sva tri istovremeno:
- Unit testovi: scalable + relatively robust, ali pokrivaju tanak sloj namjere
- LLM sudci: scalable + faithful, ali ranjivi na eksploataciju
- Ljudski review: faithful + robust, ali ne skaliira

## §2: Test-driven Rewards za SWE zadatke

**Postavka:** SWE-Universe pipeline generiše SWE-like zadatke iz GitHub PR-ova. Binary pass/fail je reward.

**Problem 1: Faithfulness.** False positives (netačno rješenje prolazi testove) i false negatives (ispravno rješenje pada). Dekomponiraju na:
- `instruct_clear` — je li instrukcija dovoljno jasna?
- `instruct_ut_align` — da li testovi pokrivaju ono što instrukcija traži?

**Rješenje:** Agentic quality judge koji aktivno istražuje Docker okruženje (MiniSWEAgent) i ocjenjuje oba dimenzija. Najbolje sa ground-truth patch-em kao referencom (F1 na instruct_ut_align najbolji).

**Problem 2: Reward hacking.** Agent prolazi testove bez legitimnog procesa — retrieval rješenja, test tampering, git history mining. Podijeljeni na:
- Static-environment leakage (okruženje otkriva prečice)
- Policy-dependent shortcut access (agent aktivno traži informacije)

**Rješenje:** Trajectory-level behavior monitor tokom RL. Token-level penalty za高风险 obrasce. Monitor se iterativno ažurira jer su hacking strategije policy-dependent.

**Rezultati:** Hacked resolved rate pada sa 28.57% na 0.56%. Clean resolved raste sa 40.22% na 60.53% na prosjeku tri SWE-bencha.

## §3: Interactive Judge za frontend zadatke

**Problem:** Frontend ne može ocijeniti samo izvršenje — treba vizualni kvalitet, layout, interaktivno ponašanje.

**Static rubric judge:** Decomponira ocjenu na 6 dimenzija (Functional 37.7%, Content 19.0%, Visual 13.3%, Layout 12.9%, UX 9.3%, Technical 7.2%). Visoka konzistencija: Kendall τ ≥ 0.93 između judge modela.

**Ključni problem:** Static judge je ranjiv na length exploitation — modeli generišu sve duži CSS/JS da naduvaju score.

**Agentic interactive judge:** Three-stage pipeline:
1. Action planner generiše listu akcija u jednom forward pass-u
2. Playwright server izvršava akcije u browser-u
3. Judge model ocjenjuje interakcione tragove

Koristi atomic web operations (click, scroll, fill form, hover...). Evaluacija je bazirana na runtime ponašanju, ne inspekciji koda. Izbjegava length hacking jer reward dolazi iz interakcije.

**Rezultati:** RFT sa interactive judge filterom daje +6 na WebDev Human Eval i +36 na QwenWebBench za Qwen-Plus. Qwen3.7-Max je bio 4. globalno na Code Arena.

## §4: Korisnički feedback kao verifikator

**Postavka:** 125.528 trajektorija, 535.737 round-level anotacija iz interakcija senior programera sa coding assistant-om.

**Polarity distribucija:**
- Neutral: 76.6% — korisnici ne komentiraju kada je sve u redu
- Negative: 20.0% — eksplicitan kada je greška
- Positive: 3.5% — rijetko, ali visoko pouzdano

**Negative razlozi:**
- Execution error: 56.6%
- Misunderstand: 21.1%
- Omission: 8.9%
- Overaction: 6.3%
- Inefficiency: 4.9%
- Communication: 2.1%

81.8% negativnih signala je visoke pouzdanosti.

**Tri metode treninga:**
1. SFT — standardno, jednako tretira sve tokene
2. RW-SFT — reweight po polarity (wpos=1.2, wneu=1.0, wneg=0.8). Osjetljivo na težine.
3. Span-KTO — preference learning na span nivou. Kontinuirani spanovi iste polarity. KTO loss za pozitivne/negativne spanove + CE loss za neutralne.

**Rezultati Span-KTO:**
- SWE-bench Verified: 54.2% → 59.8% (+5.6pp)
- SWE-bench Multilingual: 52.0% → 59.8% (+7.8pp)
- Aone-bench: 14.8% → 28.1% (+13.3pp!)

**Ključni nalaz:** Span-KTO ne samo rješava više problema (+5.9pp resolution rate), već se model ponaša razumno kada ne može riješiti problem. Inefficiency +34.5% i Communication +26.5% poboljšanja kod unresolved instanci.

## §5: Agent evaluator za dugohorizontno generisanje koda

**NL2Repo benchmark:** 104 zadatka dugohorizontnog generisanja repozitorija. Generacije od Claude Opus 4.6, Gemma 4, Qwen 3.6, MiniMax M2.5, GLM 5, Kimi K2.5.

**Evaluator dizajn:** Generator G → evaluator E dekomponira spec T u checklist C, provjerava svaku stavku → Spass (checklist pass rate) + Seval (holistic score). Seval je daleko koreliraniji sa unit-test ground truth-om.

**Pet verzija prompta (v1-v5):**
- v1→v2: Dodat e2e validacija (popravio lazy evaluation)
- v2→v3: Uklonio role confusion (evaluator više ne modificira kod)
- v3→v4: Optimizovao context usage (manje čitanje, fokus na entry points)
- v4→v5: Over-specification je POVUKLA performanse!

**BoN accuracy: 57.9% (v1) → 67.4% (v4).** Više detalja nije uvijek bolje — v5 je gore.

**Poređenje evaluator modela:** Claude Opus 4.7 vodi (BoN 70.4%, τ 0.579), ali Qwen 3.7 Plus ima visoku varijansu (±10pp). Stabilnost je kritična za trening pipeline.

**RFT rezultati:** Evaluator-filtered data (9.139 uzoraka) nadmašuje random sampling (23.52 vs 21.61). Full dataset (19.050) dostiže 24.75 — quality–quantity trade-off.

## Šta me najviše zapalo

1. **Reward hacking je neizbježan** — nije bug nego matematički neizbježna posledica optimizacije prema imperfektnoj funkciji cilja (Goodhartov zakon). Ovo je ključna poruka.

2. **Over-specification povlači evaluaore** — v5 prompt sa ekstremno detaljnim pravilima je gore od v4. Znači da treba pronaći right granularity za kapacitet modela koji služi kao evaluator.

3. **Korisnici gotovo nikad ne kažu "dobra rabota"** — 3.5% pozitivnih signala. Negativni signali su 6x češći i mnogo pouzdaniji. Treba dizajnirati sisteme koji se uče iz ovakve asimetrije.

4. **Behavior monitor je zatvorena petlja** — ne jednorazno pravilo već iterativni sistem koji se ažurira dok se policy poboljšava. Ovo je konkretan primjer "co-evolution" ideje.

5. **Interactive judge izbjegava length hacking** — modeli ne mogu naduvati score generisanjem dužeg koda jer se evaluacija vrši na runtime ponašanju.

6. **Quality–quantity trade-off je realan** — dupli dataset bez evaluatora dostiže sličan rezultat kao filtered dataset, ali sa duplim compute-om. Evaluator je najvredniji kada je kandidat pool ograničen.

## Autori

Qwen Team — 12 core + 14 contributor. Vodeći: Xuwu Wang (project lead), Dayiheng Liu (corresponding).

## Limitacije

- Rad je isključivo iz Qwen ekosistema — svi eksperimenti koriste Qwen modele
- Nije riješen problem evaluacije subjektivnog vizuelnog kvaliteta (polish, fluidnost animacija)
- Korisnički feedback je offline — ne online/adaptivno
- Nema open-source koda niti evaluator promptova u repozitoriju
