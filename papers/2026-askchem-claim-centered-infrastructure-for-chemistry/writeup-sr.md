# Pregled — AskChem: Infrastruktura centrirana na tvrdnje za sintezu hemijske literature

> **Rad:** AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis
> **Autori:** Bing Yan, Gregory Wolfe, Stefano Martiniani, Kyunghyun Cho
> **Godina:** 2026 · **ArXiv:** https://arxiv.org/abs/2607.28618

Ovo je moje lično objašnjenje, napisano nakon čitanja i ponovne implementacije
rada. Ovo je sinteza, ne prepričavanje apstrakta.

## Verzija u jednom pasusu

AskChem tvrdi da je *jedinica pretrage u naučnoj pretragi pogrešna*: i dalje
vraćamo cele radove, a istraživaču zapravo treba atomski **nalaz** — jedna tipizirana
tvrdnja sa priloženim DOI-jem izvora i doslovnim citatom. Zato oni rastavljaju 147K
hemjskih radova na 2.4M takvih tvrdnji sa provenijencijom, grade tri komplementarne
strukture (fasetnu taksonomiju, graf dokaza, "živu" taksonomiju centriranu na principe)
nad *istim* deljenim skladištem tvrdnji, i izlažu ih kroz hibridni retrieval koji
spaja četiri signala rangiranjem pomoću Reciprocal Rank Fusion. Rezultat, izmeren na
benčmarku utemeljenosti, je 100% rešivih citata i oko duplo veća gustina citata u
odnosu na sam LLM — svaki citirani DOI je stvaran jer je sintetizator *ograničen* da
citira samo DOI-jeve koji se pojave u pronađenim dokazima.

## Problem

Hemija je oblast sinteze literature. Tipično istraživačko pitanje — *"koji
elektrokatalizatori redukuju CO₂ na CO, i uz koju Faraday-jevu efikasnost?"* — ne
odgovara jedan rad; odgovara ga *skup* specifičnih nalaza razbacanih po desetinama
radova, gde svaki doprinosi jedan torku (katalizator + uslovi + izmerena vrednost).
Ali svaki alat za pretragu vraća **rangirani spisak dokumenata**. Dakle, naučnik — ili
LLM agent — mora da otvori svaki rad, pronađe relevantnu rečenicu, proveri broj i
ručno sastavi odgovor koji obuhvata više radova. To je sporo i podložno greškama.

Alternativa nije ništa bolja. Ako pustiš LLM da odgovara iz parametarske memorije, on
**izmišlja uverljivo zvučeće citate** ("Agrawal i sar., 2024") jer nema pravu predstavu
o tome koji DOI čega potkrepljuje. Tako ni pretraga dokumenata ni sam LLM ne rešavaju
utemeljenu sintezu preko više radova.

Suštinsko neslaganje koje AskChem identifikuje: **jedinica naučnog zapisa je nalaz,
ali jedinica pretrage je i dalje rad.**

## Ideja

Napravi **atomsku tvrdnju sa provenijencijom** primitivom pretrage. Svaka tvrdnja nosi
obaveznu trojku: `(tip_tvrdnje, doi_izvora, doslovni_citat)` — ili, za strukturirane
tvrdnje o celom radu koje nemaju jedan neprekidni citat, eksplicitni lokator dokaza
koji pokazuje na tabelu/sliku/sekciju. Autori koriste lepu analogiju: kao što SAM
(Segment Anything) rastavlja slike na ponovo iskoristive maske, AskChem koristi LLM-ove
da **rastavi radove na ponovo iskoristive, slive tvrdnje**.

Zatim se tri strukture nadogradjuju nad *istim* skladištem tvrdnji, sve kljucane jednim
`claim_id`:

1. **Stabilizovana fasetna taksonomija** — "o čemu je to?". Indukovana iz korpusa duž
   fiksnih pogleda (po_tipu_reakcije, po_klasi_supstance, po_primenu, ...), zatim
   stabilizovana kanonskim L1 rutiranjem + normalizacijom sinonima + fuzzy klasterovanjem
   u trajne putanje. *Operativni* indeks koji se koristi u pretrazi.
2. **Graf dokaza** — "kako su nalazi u vezi?". Tipizirane usmerene grane:
   podržava / protivreči / proširuje / izvodi_se_iz. Relacioni sloj *nad* pretragom,
   prikazan kao susedstvo oko bilo koje tvrdnje.
3. **Eksploratorna živa taksonomija** — "koji princip to upravlja?". Hijerarhija
   centrirana na principe (principi / teorije / modeli / mehanizmi) sa **mehanizmom
   uzdržavanja**: ako ništa ne pristaje, sistem *predlaže novu granu* umesto da
   nasilno uklopi. Izričito eksploratorna, nije validirana ontologija.

Pošto pretraga, pregled i navigacija grafom vraćaju objekte kljucane istim `claim_id`,
nikada ne gubiš provenijenciju dok se krećeš između njih.

## Kako radi (intuicija)

**Indeksiranje (van mreže).** Dva LLM ekstrakciona procesa, oba sa JSON-dekodiranjem
ograničenim na objekat, validacijom šeme i pokušajem iznova:
- *apstraktni ekstraktor* (GPT-5-mini, 102K radova) — visoke propusnosti, plitak;
- *duboki ekstraktor celog teksta* (Gemini 3.1 Pro preko Vertex AI batch-a, 44K radova)
  — hvata tipove tvrdnji koji nedostaju u apstraktima (ogrančenja, iznenađenja, mehanizam).

Tvrdnjama se zatim dodeljuju putanje faseti, grane dokaza i domaćini u živoj taksonomiji.

**Pretraga (na mreži) — algoritam koji nosi sve.** Ovo je najimplementabilniji deo i
tu sam se fokusirao:

1. Preformuliši upit u 3–4 ključne podupite.
2. Pokreni **četiri paralelna kanala opoziva**, svaki daje rangiranu listu tvrdnji:
   FTS5 (BM25-oliko preko teksta tvrdnje + citata), opoziv na nivou rada (rangiranje po
   autoritetu izvornog rada), opoziv preko čvorova taksonomije (poklapanje upita sa
   labeama faseta → opoziv tvrdnji) i opoziv gustim vektorom (kosinusna sličnost upita i
   ugnježdenja tvrdnji).
3. **Spoji** četiri liste Reciprocal Rank Fusion (RRF, `k=60`).
4. **Diverzifikuj** na ≤ 40 tvrdnji.
5. **Sintetizuj** LLM-om čitačem koji sme *samo* da citira DOI-jeve prisutne u pronađenim
   dokazima — ovo ograničenje je upravo ono što gura DOI postojanje na 100%.

Ključna intuicija za *zašto RRF*: četiri kanala vraćaju nesvodive sirove ocene (BM25
rezultat, broj citata, broj poklapanja, kosinus u [0,1]). Naučeni model fuzije rangova bi
zahtevao da te skale budu uporedive, što zahteva oznake. RRF koristi **samo rang**, pa je
bez parametara, bez treniranja i robustan na heterogene skale — upravo svojstvo koje
produkcijski višesignalni sistem želi. Matematika je samo:

```
RRF(d) = Σ_i  1 / (k + r_i(d))      (odsutan kanal → 1/(k+∞) = 0)
```

## Šta sam naučio implementirajući

*(Stvari koje su postale jasne tek kad sam napisao kod.)*

- **RRF je skoro sramno jednostavan.** To je oko 10 linija Python-a. Pravi posao je
  izgradnja četiri kanala opoziva tako da svaki da *rangiranu listu*; fuzija je trivijalna.
  To preusmerava rad: intelektualni sadržaj je *informaciona arhitektura* (tvrdnja + tri
  strukture), ne kombinator.
- **Invarianta tvrdnje je "DOI + tip + (citat ILI lokator)", a ne "tvrdnje imaju citate".**
  Lokator kao rezervu sam preskočio u prvom čitanju, a važan je: strukturirane tvrdnje o
  celom radu zaista nemaju neprekidan citat — pokazuju na tabelu/sliku/sekciju. Prisilno
  izvlačenje citata bi dalo smeće.
- **Validacione kapije su inženjerska suština; izbor LLM-a je sekundaran.** Iskreno
  uokvirivanje rada je da JSON-validacija šeme + provere provenijencije + pokušaj iznova
  garantuju **sledljivost**, ne semantičku ispravnost. 100% utemeljenost znači da možeš
  *kliknuti do citata* — to **ne** znači da je tvrdnja tačna. Ta razlika je cela
  epistemološka pozicija sistema.
- **Benčmark meri utemeljenost, ne činjeničnu tačnost.** Postojanje DOI-ja se izračunava
  objektivno preko CrossRef-a (bez sudije); relevantnost ocenjuje sudija sa κ = 0.914.
  Savršeno utemeljen sistem može i dalje da promaši tačan odgovor ako ga nema u korpusu.
- **Ablacija kanala je poučna.** U mojoj igrački, leksički kanali (FTS, vektor) nalaze
  zlatnu tvrdnju najbrže (srednji rang prvog zlata ≈ 1.0–1.4), dok samo-rad i
  samo-taksonomija zaostaju (2.6). Dakle autoritet i taksonomija *proširuju opoziv*;
  leksički kanali pokreću *preciznost na vrhu*. Zato je fuzija svih četiri RRF-om
  robustna — slabi kanali ne mogu da povuku rezultat nadole, ali doprinose kad leksički
  promaše.

## Šta me iznenadilo / šta je bilo teže nego očekivano

- **Nema petlje treniranja modela uopšte.** Ovo je rad o informacionoj arhitekturi /
  upravljanju podacima, ne o algoritmima učenja. Jedine "trenirane" komponente su
  zamrznuti gotov model ugnježdenja i FTS5 BM25 statistike. Zato semantika `train.py`
  mora da postane "izgradi indeks i odgovori na upite" — nema šta se gradijentno spuštati.
- **Kontrolisano poređenje je najčišći dokaz.** AskChem i Paperclip dele *isti* rewriter i
  sintetizator i razlikuju se *samo* u retrieval pozadini (tvrdnje naspram radova). Tvrdnje
  pobeđuju u relevantnosti (2.15 naspram 1.72) i na-temi stopi (86.6 naspram 57.8). To
  izoluje dobitak na **granularnost tvrdnje**, a ne na LLM — mnogo jača tvrdnja od
  "naš sistem je dobar."
- **Uzdržavanje žive taksonomije je samo po sebi signal kvaliteta.** 663 od 4.931 čvorova
  su *otvorene predložene grane*. Preferiranje "predložio bih novog domaćina" naspram
  nasilnog uklapanja je mala ali značajna epistemička disciplina; nisam očekivao da će to
  biti istaknuto kao osobina, a ne greška.
- **Diverzifikacija je neodređena.** Rad kaže "diverzifikuje na ≤40 tvrdnji" bez imenovanja
  metoda. Koristio sam pohlepni DOI/putanja limit (očigledna osnova); pravi MMR preko
  ugnježdenja tvrdnji bi bio prirodni sledeći eksperiment.
- **Nema per-kanalna ablacija u radu.** Moja leave-one-out igračka je zaista informativna, a
  je rad izostavlja — jasno otvoreno pitanje.

## Reference
- Rad: https://arxiv.org/abs/2607.28618
- Zvanični kod: https://github.com/bingyan4science/askchem · Uživo: https://askchem.org
- Moja implementacija: `implementation/` (igralište samo sa stdlib: `data.py` šema
  tvrdnje + sintetički korpus od 23 tvrdnje + mini-bench od 5 pitanja; `model.py`
  ClaimStore sa sva četiri kanala opoziva + RRF `k=60` + pohlepna diverzifikacija +
  susedstvo graf-a dokaza; `train.py` izgradi → pretraga → utemeljena sinteza → mini
  AskChem-Bench → ablacija kanala)
- Rastav: `breakdown.md`
- Beleške: `notes.md`
