# Writeup (SR) — Apodex 1.1: Scaling Agentic Intelligence for Complex Work

> Moje objašnjenje, napisano nakon dva pažljiva čitanja, raščlambe i
> from-scratch mini implementacije. Nije rezime apstrakta — ovo je šta zaista
> mislim da se dešava, plus šta me naučio kod.
>
> **Rad:** Apodex Team, *Apodex 1.1: Scaling Agentic Intelligence for
> Complex Work*, arXiv:2608.23283 (2026).

## Verzija u jednom pasusu

Apodex 1.1 je sistemski izveštaj frontir laba (397B model + 35B Mini) o jednoj
tezi: agentnu inteligenciju treba meriti *završenim poslom*, ne tačnim
odgovorima, i skalira se istovremeno duž dve ose. **Skaliranje okruženja**
čini File/Search/Code svetove izvršivim i *verifikabilnim po ugovoru*, tako
da svaka akcija, oporavak i artefakt postaju trening signal. **Skaliranje
koordinacije** trenira sam model da razlaže, delegira, integriše asinhrone
rezultate i replanira (Agent Team runtime). Zajednički runtime (**AgentOS**)
drži autoritativno stanje workspace-a, namespacove, publication lease i
replayable manifeste, a sve se trenira SFT-om plus **PIVOT-RL**-om —
hindsight lokalizacijom pivotalnih loših odluka u hintovane, state-restored
nastavke zadataka. Vodi na finance/science/search vertikalama i
takmičarskoj matematici, srednji je paket na profesionalnom artefakt radu, i
jasno zaostaje na kodiranju. Dva najportabilnija artefakta su formalizam
task contracta i PIVOT-RL; oboje sam reimplementirao u minijaturi, i
PIVOT-RL *nije* ubrzao učenje na toy skali — detalji ispod.

## Problem

LLM-ovi umeju da odgovore, ali ne mogu pouzdano da *rade*. Pravi posao je
long-horizon: nađi dokaze, operiši po fajlovima, pokreći i debaguj kod,
održi plan koherentnim kroz desetine koraka, primi ljudsku poruku usred
posla, oporavi se od greške bez bacanja validnog progresa, i preda artefakt
koji neko može da proveri. Benchmarki koji mere jedan prompt → jedan odgovor
propuštaju upravo modalitete grešaka koji dominiraju u produkciji: drift
stanja, izgubljena provenance, neverifikabilni deliverables. Papirova
jedinica — "working capability" — čini *krajnje stanje workspace-a*, ne chat
transkript, proizvodom.

## Ideja

Tri dela, jedan sistem:

1. **Formalni task contract** (Eq. 1–5). Izvršivi zadatak je torka
   `E = (W, W0, q, A, T, Ω, B, D, V_D)`: svet, početno stanje,
   normalizovani cilj, akcije, dinamika, observacije, *budžet vektor*,
   delivery contract i terminalni verifikator. Presuda dolazi iz
   `S_D = V_D(W0, W_H, τ_H)` — sudi ti se po predatakom workspace-u plus put
   kojim si došao. Verifier-as-tool (usred runa, proizvodi observacije) je
   konceptualno odvojen od terminalnog verifikatora. Replay zahteva
   nepromenljiv environment manifest (seed, verzije alata, log akcija).
2. **Agent Team + AgentOS.** Tim razlaže na eksplicitni task board (unosi sa
   resolution statcima; `resolved` znači *provereno*, ne *završeno
   procesom*), delegira workere, verifikuje asimetrično (napadni JEDAN
   claim, nemoj rešavati ponovo) i sintetizuje iz evidence grafa. AgentOS je
   runtime ugovor ispod: state torka `W_t = (F, Q, C, I, G, K)`, namespaco-
   vi (`/inputs`, `/workspace`, `/outputs`, `/shares`), capability-scoped
   pristup, tiered compaction i — moj omiljeni mehanizam — **controlled
   delivery**: objavljivanje deklariše eksaktni manifest output putanja,
   run-scoped lease dozvoljava najviše jednog committera, i svaki unos
   manifestа se pomiri sa baseline snapshotom, prazan ili stari fajl ne može
   da lažira dostavu.
3. **PIVOT-RL** (jedini konkretni algoritam). Retrospektivna analiza
   neuspelih trajektorija pronalazi *pivotе* — korake gde je politika
   izabrala neplodnu strategiju, zloupotrebila alat ili nikad revidirala
   pretpostavku. Zadrži koristan prefiks, vrati stanje okruženja, dodaj
   kratki korektivni hint (samo pravac, nikad meta predikcije, odsutan na
   inferenciji) i treniraj na nastavku. Mešaj hintovane nastavke sa
   nehintovanim punim zadacima.

Evaluaciona filozofija je tiho genijalni deo: svaki benchmark pokreće *isti
model* na dva načina — ReAct (minimalan scaffold, izoluje politiku) vs Agent
Team (trenirana koordinacija + organizovan compute). AT−ReAct jaz *jeste*
izmerena vrednost skaliranja koordinacije: +9.3 GDPVal, +12.1 search, +16.7
proof checking, +11.8 BioMystery.

## Kako radi (intuicija)

Zamisli tri sata koji se okreću različitim brzinama. **Sat okruženja**:
svetovi se prave "forward jeftino, inverzija skupo" — generator definiše
latentno poslovno stanje, odnose autoriteta i logiku izvođenja, pa ih
*projektuje* u neuredan workspace; generator svoj zadatak rešava jeftino iz
latentnog stanja, ali agent mora da rekonstruiše put iz renderovanih fajlova
pod budžetom. Težina nije veličina direktorijuma; nego
`ρ_acq = (N_cand + N_hop) / B_tool` — koliko *nužne inspekcije* budžet
mora da apsorbuje. **Sat koordinacije**: board, asinhrone intervencije (12
treniranih kategorija — politika odlučuje šta intervencija invalidira, a šta
preživljava) i asimetrična verifikacija teku kao obični tool pozivi unutar
istog contracta; koordinator poseduje semantiku boarda, runtime status
izvršenja, i board edit ne može da pregazi live job. **Sat treniranja**:
greške se vraćaju kroz "managed self-evolution" petlju — rekurentni modaliteti
grešaka postaju spec za sledeću rundu konstrukcije okruženja. Ništa
mistično, i bez self-modifikacije modela; inženjerska petlja u kojoj
verifikacija odlučuje šta je dovoljno pouzdano za trening.

## Šta sam naučio implementacijom

Moja implementacija (`implementation/`, ~2 min CPU) pravi **ChainWorld** —
toy koji instancira contract (Eq. 1–8), mini AgentOS (namespacoVi, lease,
manifest, deterministički replay — selftestovan), PIVOT-RL sa hindsight
detekcijom pivotа i tri trening kraka (`full`, `pivot`, `randcut` — potonji
seče nastavke na *nasumičnoj* tački istih grešaka, čime izoluje "seči na
failure-relevant stanju"), i skriptovani Agent Team. Stvari koje su se
razbistrile tek uz kod:

- **Contract je lakših 80%.** NamespacoVi + lease + exact-manifest
  objavljivanje + baseline pomirenje su ispali iz Eq. 1–7 formalizma skoro
  mehanički. Kad jednom "delivery" postane first-class objekat sa
  verifikatorom koji vidi `(W0, W_H, τ_H)`, većina nemara u agentskim
  evaluacijama postaje strukturno nemoguća, a ne stvar discipline.
- **"Forward jeftino, inverzija skupo" je stvarno dobar recept za
  generatore.** Prvo sam gradio latentni lanac signala, projektovao ga u
  fajlove sa distraktorima, i podešavao ρ_acq budžetom. Ekshaustivna
  pretraga traži 17 tool poziva; praćenje provenance 2L+1 = 9; budžet je 14.
  Naučena politika konvergira tačno na `R0 C0 R4 C4 R1 C1 R6 C6 SUB` — oblik
  rešenja samog generatora. To je bilo zadovoljstvo gledati.
- **ReAct↔AT poređenje traži compute matching, a papir ga ne radi.** Moj
  skriptovani tim takođe pobeđuje (100% SR) — ali troši 18 organizovanih
  tool poziva naspram solo budžeta od 14. Lift je realan ali kontrola
  fali, i na toy skali i (sada sumnjam) u papiru.
- **Reward shaping je bio neizbežan na toy skali.** Sa čistim terminalnim
  rewardom, random politika od 25 akcija nikad ne dovrši lanac od 9 koraka,
  pa gradijent ne postoji. Koristio sam verifier-izveden shaping (progres
  ka contractu, distraktori ostaju fatalni). 397B model ima jezičke priore;
  moj pointer transformer nema. Vredi znati na kojoj strane te linije je
  svaki pokušaj replikacije.

## Šta me iznenadilo / bilo teže nego što sam očekivao

- **PIVOT-RL NIJE ubrzao učenje u mom toy-u — pošten negativan rezultat.**
  Epizoda do prvog ≥90% eval SR: `full` 1920, `pivot` 3249, `randcut` 4160.
  Po wall-clock rundama krakovi su skoro izjednačeni (120 / 140 / 130 rundi)
  jer pivot krak troši pola epizoda na nastavke. Verujem da mehanizam ipak
  važi — na horizontu od 14 koraka restart neuspelog zadatka košta skoro
  ništa, dok papirovi dobitci dolaze iz long-horizon LLM zadataka gde su
  restarti skupi a failure-relevant stanja retka. Ali ne mogu to da pokažem
  na ovoj skali, i hteo bih Fig. 4 iz papira raskomadan po dužini horizonta
  pre nego što u potpunosti kupim priču.
- **Disciplina hintova se ipak replikuje**: pivot krak se evaluira bez
  hintova i dalje dostiže 100% — direkcioni hintovi nisu postali štaka.
- **Headline brojke su tražile drugo čitanje.** Apstraktno "leading
  performance band" važi samo za finance/science/search vertikale i
  takmičarsku matematiku. Na neutralnom terenu zaostaje: SWE-bench Verified
  77.7 vs Opus-5 92.2; Terminal-Bench 2.1 70.8 vs 91.9; HLE 56.1 vs 64.7.
  Deo baseline unosa su Apodexove sopstvene reprodukcije pod Apodex
  harnessom. Interni benchmarki (FSB, FRB) su kućni teren. Čitati tabele pre
  apstrakata ostaje jedini način da se ostane pošten.

## Verdikt

🏗️ Sistemski izveštaj, ne methods papir — čitaj ga zbog **task contract +
AgentOS runtime ugovora** (najportabilniji spec za verifikabilna agentska
okruženja koji sam video od rada o harness skaliranju) i zbog two-view
ReAct/AT dizajna evaluacije. PIVOT-RL je jedini algoritam vredan krađe, uz
ogradu da moj toy nije reprodukovao ubrzanje i da se preciznost detekcije
pivota nikad ne ablatira u papiru. Tvrdnje o "vodećem bendu" tretirati kao
ograničene na vertikale.

## Reference

- Papir: https://arxiv.org/abs/2608.23283
- Zvanični repo: https://github.com/ApodexAI/FrontierAgent
- Moja implementacija: `implementation/` (README sa poštenim tabelama
  rezultata)
- Raščlamba: `breakdown.md`
- Beleške (oba čitanja): `notes.md`
