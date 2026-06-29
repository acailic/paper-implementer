# Writeup — JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting

> Evo kako bih ovo objasnio prijatelju pre piva, ako me pita "šta si čitao?"

> **Jezici:** [English](writeup.md) · Srpski (ovaj fajl)

---

Jednostavna priča ide otprilike ovako.

Kad veliki jezički model generiše tekst, radi to jedan token po jedan. To je sporo. Spekulativno dekodiranje je trik da se to ubrza: mali model pogodi sledećih nekoliko tokena, pa se svi ti pogođaji paralelno provere protiv velikog modela. Ako su pogođaji dobri, napreduješ nekoliko tokena u jednom koraku umesto jednog. Što brže možeš da pogađaš i što češće budeš u pravu, to je veće ubrzanje.

Problem je što su postojeće metode udarile u zid oko 4-6× ubrzanja. Razlog je ono što autori zovu "dilema kauzalnost-efikasnost." Imaju dve porodice pristupa, i svaka žrtvuje jednu stvar da bi dobila drugu.

Autoregresivni drafteri kao EAGLE prave zaista dobre pogođaje — svaki token je uslovljen prethodnim, baš kao što to radi target model. Ali oni su sekvencijalni: da bi nacrtao stablo dubine 16, treba 16 odvojenih forward prolaza. To je skupo. Bidirekcionni drafteri kao DFlash rade sve u jednom prolazu — super jeftino po tokenu — ali svaka pozicija je predviđena nezavisno bez znanja šta je došlo prije na toj konkretnj grani. Znači možeš da dobiješ stablo gde je najbolja grana "given told that" — oba su riječi plauzibilna sama za sebe, ali nitko ne kaže "given told" na engleskom. Stablo troši budžet na grane koje izgledaju dobro u izolaciji, ali se ruše kad se verifikuju zajedno.

JetSpec-ova ideja je mrtvo prosta kad je jednom vidiš: stavi kauzalnu attention masku na paralelni draft head. Svaka pozicija u stablu može da prisustvuje prefiksu i vlastitim precima, ali ne potomcima ni sestrinskim granama. Sve pozicije se i dalje računaju u jednom forward prolazu — dakle jeftino — ali svaka grana sad ima prave kauzalne zavisnosti. Draft distribucija kopira autoregresivnu faktorizaciju target modela, samo paralelno.

## Dve stvari koje su me stvarno iznenadile

Prvo — **failure mode je dramatičan i strukturan.** Papir ima ovu prekrasnu studiju slučaja na MATH-500 prompt 0. Diffusion head-ova najbolja grana je "given told that", koja ima draft score od −3.76 nats ali target-model zajedničku vjerojatnost od −63.32 nats. To je e⁻⁶³ — funkcionalno nulta vjerojatnost. Grana kombinuje dvije riječi koje su individualno plauzibilne na svojim dubinama, ali se ne mogu slijediti jedna drugu. Zaista koherentna "are told that" sjedi na rank 3. Kauzalni head? Njegova rank-1 grana se slaže sa target-om unutar −0.34 nats.

I ovo nije izuzetak. Preko 50 promptova, diffusion-ov rank-1 gap je veći od kauzalnog na 92% promptova, sa median gapom 5× većim. Kauzalni head ne treba nikakve loss-weighting trikove da bi ovo popravio — sama struktura maske sprječava failure mode. To je inženjersko svojstvo, ne samo broj na tabeli.

Drugo — **po-token trošak draftovanja je apsurdno nizak.** Appendix G profilira stvarni hardverski trošak na H200. Pri draft dubini 256 sa context length 1024, po-token trošak je 0.054% od jednog target verification prolaza. To je u "ultra-jeftinom" režimu iz papir-ove vlastite teoretske analize. Citav scaling argument leži na ovome: ako je draftovanje skoro besplatno, jedino što važi je kvalitet akceptacije. I to je tačno ono što kauzalno tree draftovanje pruža.

## Šta sam naučio o dizajn prostoru

Ablacije su u ovom papiru neuobičajeno čiste i svaka kaže nešto specifično.

**Reverse KL distilacija je katastrofalna za tree draftovanje** — 36-46% relativni pad u poređenju sa forward KL. Razlog: reverse KL je mode-seeking, koncentriše vjerojatnosnu masu na top-1 predikciju. Ali tree draftovanje treba raznolikost po granama. Ako draft head uvijek predviđa isti top-1 token, stablo nema korisnih grana za istraživanje. Forward KL čuva target-ovu soft-label distribuciju, drži više plauzibilnih nastavaka živim. SFT je negdje između — radi, ali nije dobar kao forward KL.

**Loss weighting γ je štapić za diffusion head-ove.** DFlash-ov trening objektiv koristi eksponencijalni decay weight po poziciji — pozicije daleko od anchor-a doprinose manje loss-u. Diffusion head je ekstremno osjetljiv na ovo: speedup skače sa 5.46× na 8.16× na 6.17× kroz γ=0, 7, 15. Kauzalni head? 8.29, 8.50, 8.41 — u suštosti ravno. Kauzalna maska strukturno sprječava problem inkonzistencije, pa ne treba da štucaš weight raspored da bi ga popravio.

**Entropy-guided tree konstrukcija kolapsira.** Inicijalno sam mislio da prioritetizacija visoko-entropijskih pozicija (nepouzdanih predikcija gdje je model "znatizavezan") može pomoći da se istraže raznolike grane. Nop — 4.76× speedup naspram 8.15× sa kumulativnim log-probom. Entropija sama ne kaže koje su grane zapravo vjerojatne; samo kaže gdje je model nesiguran. Grana koja je nesigurna na svakoj dubini vjerovatno nije dobar nastavak. Kumulativni log-prob tačno identifikuje grane koje su zajednički vjerojatne.

**Budžet i batch size se trguju u serving-u.** vLLM integracija priča jasnu priču: budžet 256 daje 7.58× ubrzanje na batch size 1, ali pada na 2.85× na batch size 16. Velika stabla pomažu kad serviraš jedan request, ali tree verification overhead sam postaje bottleneck kad batchiraš mnogo requestova. Pravi budžet zavisi od tvoje serving opterećenosti.

## Brojevi ubrzanja

Ovo je onaj papir gdje abstract brojevi zvuče previše dobri dok ne pogledaš tabele i videš da drže preko svih benchmaraka:

```
ubrzanje (×)
10 │  · JetSpec MATH-500 (9.64×)
   │  · JetSpec AIME25 (10.76×)
 8 │     · JetSpec HumanEval (9.95×)
   │       · DDTree MATH-500 (8.78×)
 6 │  · JetSpec MT-Bench (7.67×)
   │
 4 │
   │          · EAGLE-3 MATH-500 (2.35×)
 2 │
   │
 0 └──────────────────────────────────────
     matematika    kodiranje    chat
```

Konzistentan gap između JetSpec-a i DDTree-a (otprilike 10-15% viši speedup na budžetu 256) je u potpunosti objašnjen kauzalnim conditioning-om: isti tree budžet, isti hardver, isti trening podaci — samo bolja maska.

## Šta je bilo teže nego što sam mislio da razumijem

Tree-kauzalna attention maska zvuči prosto ali implementation detalji su važni. Kad imaš više blokova (svaki sa anchor + future pozicijama), maska mora da osigura da (a) pozicije unutar istog bloka vide ranije pozicije u tom bloku, (b) pozicije vide puni verifikovani prefiks, i (c) pozicije iz različitih blokova se ne međusobno vide. Papir-ova Figura 5 pokazuje ovo jasno — nije standardna kauzalna maska, nego blok-kauzalna maska sa dijeljenim prefiksom. Da ovo pogrešiš, ti tiho slomi kauzalno svojstvo bez ikakve očigledne greške.

vLLM integracija je netrivijalni inženjering. Tree verifikacija zahtijeva custom attention kernel koji gradi odnos predaka za speculative nodove, primjenjuje tree masku unutar attention-a, i verifikuje sve kandidate bez materializovanja dense per-request maske. Implementirali su ovo kao fused paged tree-attention kernel koristeći NVIDIA CuTe DSL na SM90. To nije nešto što samo copy-paste-ujem.

## Reference
- Papir: https://arxiv.org/abs/2606.18394
- Zvanični kod: https://github.com/hao-ai-lab/JetSpec
- Projektna stranica: https://jetspec-project.github.io/jetspec-web/
- Breakdown: `breakdown.md`
