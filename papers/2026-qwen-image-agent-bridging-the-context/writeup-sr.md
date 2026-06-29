# Writeup — Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation

> Evo kako bih ovo objasnio prijatelju pre piva, ako me pita "šta si čitao?"

> **Jezici:** [English](writeup.md) · Srpski (ovaj fajl)

---

Jednostavna priča ide otprilike ovako.

Zamoliš text-to-image model da "nacrta scoreboard za NBA finale 2026. sa logima oba tima i ukupnim rezultatom serije." Model pokuša. Ne zna ko je igrao. Ne zna rezultat. Ne zna kako izgledaju logoi. Halucinira nešto uko loose košarkaški oblik i prelazi dalje.

Ovo nije problem renderovanja. Model može da crta fino kad mu daš sve što treba. Problem je što mu *nisi dao sve što treba* — a model nema načina da traži više.

Ovaj rad naziva taj jaz **Context Gap** — neslaganje između onoga što korisnik kaže i onoga što generator slika zapravo treba da obavi posao. I grade sistem koji taj jaz premošćuje.

## Ideja u jednom pasusu

Umesto da tretiraš korisnikov prompt kao konačni ulaz za generator slika, tretiraj ga kao *početnu tačku*. Agent pogleda prompt, shvati šta fali, izađe i nađe nedostajuće komade (razmišljanjem, pretragom interneta, čupanjem iz memorije ili samo-ispravljanjem), sastavi sve u bogat detaljan prompt, i *onda* ga preda generatoru. Generator postaje renderer na kraju pipeline-a, ne čitav sistem.

## Pipeline — tri nivoa planiranja, četiri izvora konteksta

Sistem ima dva glavna dela. **Context-Aware Planning** radi na tri nivoa:

1. **Informativni nivo:** "Šta ne znam?" — identifikuje jazove, postavlja pitanja, usmjerava ih ka pravoj strategiji
2. **Sadržajni nivo:** "Sada napiši kompletni spec" — sastavlja prikupljene informacije u detaljan prompt sa subjektom, atributima, layout-om, stilom i tekstom
3. **Generativni nivo:** "Kako ovo rasporedim preko više slika ili rundi?" — radi sa multi-image i multi-turn scenarijima

**Context Grounding** popunjava te jazove iz četiri izvora:

| Izvor | Primjer |
|-------|---------|
| **Razmišljanje** | "CN Tower je znamenitost Toronta" — VLM zaključi |
| **Pretraga** | "Knicks pobijedio Spurs 4-1 u NBA finalu 2026" — izguglano |
| **Memorija** | "Korisnik preferira akvarel stil" — pamti iz prve runde |
| **Feedback** | "Samo 4 crvena auta, ne 5" — samo-provjereno i ispravljeno |

Cijela stvar je bez treniranja. Omata se oko bilo kojeg postojećeg generatora slika. U ovom slučaju koriste Qwen-Image-2.0 za generaciju i GPT-5.5-0424 kao mozak koji radi sav planning i razmišljanje.

## Benchmark — IA-Bench

Rad takođe donosi benchmark, što je dobro jer bez toga ne možeš zaista ocijeniti da li bilo šta od ovog ima smisla. IA-Bench pokriva četiri kapaciteta preko 17 podzadataka i 730 test instanci:

- **Plan** — kompozicija, enumeracija, multi-panel layouti, lavirintni putovi
- **Razmišljanje** — matematika, nauka, zdravorazumsko, mape, geometrija
- **Pretraga** — IP likovi (igre, filmovi, anime, poznati), stvarni info (berze, vrijeme)
- **Memorija** — korisnički profili, historija konverzacije preko rundi

Evaluacija koristi VLM sudije koji provjeravaju po detajjnim checklistama. Dvije metrike: Pass Rate (svi itemi moraju proći, strogo) i Checklist Accuracy (prosjek udovoljenosti).

## Dvije stvari koje su me stvarno iznenadile

Prvo — **uklanjanje pretrage ne samo slabi pretraživačke zadatke, nego ih uništava.** Ablacija pokazuje da Search PR pada sa 46.1 na 7.8 kad ukloniš modul za pretragu. To nije postepeni pad, to je litica. Ima smisla u retrospekciji — ti zadaci se uopšte ne mogu riješiti bez eksternog znanja — ali je razmjer zapanjujući. Znači da pretraga nije "nice-to-have", već nosivi zid za oko trećinu real-world zadataka generisanja slika.

Drugo — **MLLM backbone znači više nego sam očekivao.** Zamjena GPT-5.5-0424 sa Qwen-Plus ruši IA-score sa 45.4 na 19.3. To je kolaps od 57% samo od promjene planera, ne renderera. Cijela inteligencija sistema živi u planeru — on je taj koji identifikuje jazove, usmjerava upite, sastavlja kontekst, piše detaljne prompote. Kad planer oslabi, renderer nikad ne vidi dobre ulaze. Ovo je najjači argument u radu zašto agentički okvir ima smisla: model za renderovanje se jedva promijenio, ali performanse sistema su se ogromno promijenile jer se promijenila *konstrukcija konteksta*.

## Šta mislim o feedback petlji

Iskreni nalaz je da feedback dodaje najmanje vrijednosti. Uklanjanje ga samo spušta IA-score sa 45.4 na 42.1. Dva razloga: Qwen-Image-2.0 je već jak renderer pa nema mnogo za ispravljati, i VLM feedback je generički (nije task-specifičan). Autori to otvoreno priznaju i predlažu da feedback u budućnosti treba gurati ranije u pipeline (da nadzire identifikaciju context gapa, ne samo post-hoc kritiku). To je pravi instinkt — čekanje poslije generisanja da se stvari poprave je inherentno ograničeno kad je problem bio u promptu.

## Granica razmišljanje vs pretraga

Ovo je zaista zanimljiv dizajn problem. Neke činjenice se mogu riješiti LLM-ovim parametrijskim znanjem ("CN Tower je u Torontu") a neke zahtijevaju web pretragu ("koji je bio rezultat 14. augusta 2025?"). Gdje povučeš crtu? Odgovor iz papira: parametrijsko za zdravorazumsko, pretraga za precizne činjenice (tačne brojeve, datume, imena) i dinamičke činjenice (stvari koje se mijenjaju kroz vrijeme). To je čisto i principijelno. Ali priznaju da zavisi od MLLM-ovog znanja — kako bazni modeli postaju pametniji, više stvari se seli iz "treba pretragu" u "može se razmisliti o tome". Granica je živa.

## Šta me smeta

**Nema koda.** Ovo je rad o framework-u sa detaljnim opisom pipeline-a i bez ijedne implementacije. Za agentički sistem gdje inženjerski detalji (kako tačno rutiraš pitanja? kako pamtiš memoriju? kako izgleda DAG?) imaju ogromnu važnost, ovo je pravi jaz. Ne možeš to inspektovati, ne možeš reprodukovati, ne možeš graditi na tome.

**SOTA brojevi su djelimično efekt backbona.** Sistem koristi GPT-5.5-0424 + Qwen-Image-2.0, oba state-of-the-art zatvorena modela. Ablacija pokazuje da zamjena na slabije alternative kolapsira performanse. Pa koliko od "45.4 IA-score" je framework, a koliko je samo to što imaš najbolje alate u šupi? Framework jasno pomaže (17.4 → 45.4 naspram golog Qwen-Image-2.0), ali apsolutni brojevi su napuhani kvalitetom proprietarnog backbona.

**Latencija.** Kompletni pipeline je mnogo skuplji od one-shot generisanja. Višestruke LLM pozive za planning, web API pozive za pretragu, VLM pozive za feedback, plus sama generacija. DAG izvršavanje pomaže sa paralelizmom ali ne može eliminisati sekvencijalne zavisnosti. Za real-time ili cost-sensitive aplikacije, ovo je ozbiljno ograničenje.

## Glavni zaključak

Context Gap frejming je pravi doprinos ovdje. Daje ime i strukturu nečemu što svi u polju osjećaju ali nisu formalizovali: T2I modeli ne uspijevaju u realnom svijetu ne zato što ne mogu crtati, nego zato što ne znaju dovoljno o tome šta da crtaju. Agentički pipeline je razuman pristup za premošćivanje tog jaza — a IA-Bench rezultati dokazuju da radi. Ali dok se kod ne otvori i zavisnost od backbona ne smanji, teško je znati koliko od ovoga je dizajn framework-a a koliko je brute-force kvalitet modela.

## Reference
- Papir: https://arxiv.org/abs/2606.26907
- Breakdown: `breakdown.md`
