# Pisano djelo — Alaya-EVOKE: From Linear-Scaling Supervision to Endless World

> **Rad:** Alaya-EVOKE: From Linear-Scaling Supervision to Endless World
> **Autori:** Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, Feng Zhao
> **arXiv:** 2608.13546 — https://arxiv.org/abs/2608.13546
> Napisano nakon dva pažljiva čitanja i vlastite reimplementacije četiri
> ključna mehanizma u `implementation/`.

## Verzija u jednom pasusu

Interaktivni world model treba da generiše video bez kraja i da pritom
sluša korisnika usred generisanja. Dve stvari to blokiraju: ako se istorija
drži u kontekstu denoisera (ili KV kešu), cena raste sa dužinom sesije; i
student sa par koraka (distilovan za latenciju) može biti najviše onoliko
stabilan koliko njegov učitelj. EVOKE problem deli na dvoje: **memorija
postaje geometrija u eksternoj "world state bank" indeksiranoj po kameri**
(dovlači se samo ono što trenutni kadar vidi; kontekst ostaje ograničen), a
**stabilnost postaje svojstvo horizonta supervizije učitelja** (sparse
attention učitelj koji cenovno prihvatljivo supervizuje 30-sekundne prozore i
distiluje se u 3-koraknog, CFG-free studenta preko distribution matchinga
na self-forced rolloutima). Rezultat: sesije bez kraja — 65-minutni
rollouti ostaju fotometrijski ravni — po 2.11 s wall-clock vremena na
1.5-s chunk, sve na jednoj H200.

## Problem

Interaktivni world modeli imaju tri zahteva koji se međusobno tuku:

1. **Perzistentna memorija.** Ako korisnik okrene kameru 180° posle deset
   minuta, scena iza njega mora i dalje da postoji. Standardni trikovi —
   konatenacija frejmova u kontekst denoisera, rastući KV keševi — skaliraju
   sa dužinom sesije, pa svaki realni sistem seče istoriju i tiho zaboravlja.
2. **Odzivnost.** Interakcija traži malo denoising koraka. Ali few-step
   studenti se distiluju, a destilacija je ograničena učiteljem: ako
   učitelj ume da oceni samo kratke prozore, drift koji je lokalno
   prihvatljiv a globalno pogrešan nikad ne proizvede trening signal.
3. **Konzistentnost na duge staze.** Dve sorte: *degradacioni drift*
   (creep ekspozicije, propadanje teksture — vidljiv u lokalom prozoru ali
   uzrokovan ranije) i *sadržajni drift* (identitet scene se mijenja dok
   svaki kratki prozor izgleda dobro — identifikuje se tek kad se udaljeni
   trenuci porede *zajedno*).

Ključno uokvirenje rada: ova dva drifta imaju **različite vlasnike**.
Degradacioni drift je problem temporalne evolucije → rešava se dugim
*horizontom supervizije*. Sadržajni drift pri povratku na scenu je problem
prostornog setanja → rešava se *skladištenom geometrijom*. Duži prozor
supervizije ne može da oživi opažanje koje je izašlo iz konteksta; banka
memorije ne može da odluči kako se sadržaj koji nikad nije viđen razvija.

## Ideja

Tri spojena mehanizma:

**1. Rekurentna sesija sa eksternim stanjem (Eq. 1).** Sesija je petlja po
1.5-s chunkovima (9 latentnih frejmova):

```
r_k     = Read(M_k, P_k)              # renderuj geometriju relevantnu za trenutnu kameru
x_k     ~ p_theta(. | r_k, h_k, c_k)  # generiši chunk iz rendere + ograničene istorije + teksta
M_(k+1) = Write(M_k, x_k, P_k)        # depth-unprojektuj nove frejmove, dodaj u banku
```

I `h_k` (tiered istorija denoisera, samo ~3.2 s / 19 latentnih frejmova) i
`M_k` (banka, kap od 90 s ubrane geometrije ≈ 720 izvornih frejmova) rade
pod **fiksiranim budžetima** — tako da cena jednog rekurentnog poziva ne
zavisi od toga koliko dugo sesija traje. Beskonačnost = više rekurentnih
poziva, ne veći pozivi.

**2. World state banka indeksirana po kameri.** Write putanja pušta
monokularni depth model na 12 frejmova svakog generisanog chunka,
unprojektuje u world space po poznatoj trajektoriji kamere i dodaje u
banku — namerno **bez fuzije između chunkova** (fuzija je odakle dolaze
scale drift i šavovi na dugim sekvencama). Read putanja je čisto
geometrijska, bez naučenog retrivala: rangira izvorne poglede po
**co-visibility** sa ciljnim pozom, uzima do 8 dovoljno različitih,
renderuje sa z-bufferom u view-aligned warp + per-pixel visibility masku.
Nepodržani pikseli dobijaju šum σ = 1 (ne nose vizuelnu informaciju);
vidljivost svedena po patch rezoluciji izbacuje nepodržane *istorijske*
tokene iz denoisera. To je RAG, ali ključevi su pozicije kamere a
dokumenti geometrija — pravi izbor, jer se setaš onoga što je opaženo, a
ne zaključuješ ono što nije.

**3. Duga supervizija po linearnoj ceni.** Učitelj (Wan2.2 A14B DiT, sa
high/low-noise ekspertima) dobija **sparse chunk attention**: svaki query
chunk gleda first-frame sink, lokalne susede, kompresovane obližnje
frejmove, M importance-selected udaljenih frejmova i linear-attention
globalno stanje. Ograničeni ključevi po query-u ⇒ linearna cena po dužini
sekvence ⇒ supervizija rollouta od ~31 s (GT prefiks + 20 self-forced
studentskih chunkova, 189 latentnih frejmova) postaje dostupna. Destilacija
je DMD-style distribution matching: učitelj i kritik **dele jedan backbone**
(LoRA prekidač bira ko), gradijent gura studentski uzorak ka distribuciji
učitelja sa data-skaliranim normalizatorom ν, maska Ω izbacuje prefiks i
prvi generisani chunk (zaštita od flickera na granici), i — trik vredan
krađe — **istorija je detached između rollout chunkova**: svaki chunk ima
nezavisan backward graf, pa se ceo 31-s prozor *ocenjuje* zajednički dok
aktivaciona memorija ostaje chunk-size. Per-chunk tekst kondicioniranje
(captioni seku svakih 12 s i mapiraju na chunkove) čini promenu prompta
usred sesije pravim *interfejsom* — to je "evocation".

## Kako radi (intuicija)

Zamisli šta student sme da nauči. Windowed objective
`L_W = E D(q_θ(k:k+W−1), p(k:k+W−1))` daje **nula gradijenta** za sve što
nije izraživo u W-prozorskoj statistici. Skrati W i nimalo treninga neće
naučiti studenta da mu ekspozicija creepuje, jer je uzrok *pre* prozora.
Proširi W i student se supervizuje na široj distribuciji sopstvenih,
rollout-poremećenih istorija — a to je tačno distribucija u kojoj mora da
preživi na inferenciji. Pod self-forced rolloutima ovo je on-policy
distilacija: učitelj ne ocenjuje uvežbane trajektorije, ocenjuje
studentovo *sopstveno* lutanje.

A banka je komplement: kad se kamera vrati na poz, tekst kondicioniranje
ne može (i ne treba) da prizove ono što je viđeno pre 60 s — geometrija
može. Papirin timed-control eksperiment to razdvaja oštro: tekst klauza
usred sesije se realizuje u 67% slučajeva kad cilja *slobodan* sadržaj
(ceiling 83%), ali samo 4% kad bi zahtevala *prepisivanje geometrije
zaključane u banci* (floor 17%). **Tekst upravlja slobodnim sadržajem;
skladištena geometrija se opire prepisivanju.** To nije bag — to je
produktna odluka.

## Šta sam naučio implementirajući

(Iz toy reimplementacije u `implementation/`; sve ispod je *izmereno*,
brojevi u `implementation/results.json`.)

- **Ravnost je svojstvo saturacije, ne asimptota.** U E1 sam pustio
  100-chunk rekurentne sesije i gledao kako se aktivni izvorni pool banke
  popunjava do budžeta (~60 izvora na toy 90-s retenciji) — posle čega je
  vreme po koraku ravno (19.9 ms vs 19.8 ms u kasnim prozorima, odnos
  0.98). Pre saturacije cena *zaista* raste. Papirina tvrdnja o ravnoj
  ceni je zapravo "budžet ograničava rad, a budžet se popuni brzo".
- **Recall je bukvalno `retencija ≥ vreme_odlaska`.** E2 je sweep-ovao
  budžet retencije protiv povratka posle 60 s: 15 s → 0 dB (ništa),
  45 s → 12.2 dB sa 24% pokrivenosti (delimično), 90 s → 14.6 dB sa 98.6%
  pokrivenosti. Papirinih 20/21 poređenja po adresi poza koja prate
  predviđenu tranziciju je tačno ono što toy pokazuje — to je *prekidač*,
  ne gradijent, jer su izvorni frejmovi ili u banci ili nisu. A plato
  (~15 dB u mom toy-u, 15.4–17.8 dB u radu) kaže da je setanje
  semantičko, ne fotografsko.
- **Zašto je bez fuzije pravi default.** Moja Write putanja samo
  unprojektuje-i-dodaje. Čim razmisliš o fuziji geometrije dva chunka,
  nasleđuješ relativnu grešku poza kao akumulirani scale drift — a
  z-buffer na Read vremenu već razrešava preklapanja. Fuzija bi dodala
  failure mode da kupila estetsko poboljšanje koje budžet čini
  nepotrebnim.
- **Pobeda sparse attentiona je strukturna, ne implementaciona.** E3 je
  izmerio kontekst po query-u na 22 tokena *bez obzira na dužinu rollouta*
  (state + sink + lokalno + odabrani udaljeni), a analitički FLOPs od 8→128
  chunkova rasao ×22 (sparse) vs ×256 (dense). Kad su ključevi-po-query
  ograničeni, linearnost je aritmetika, ne inženjering.
- **Horizont se prenosi, i prenosi se *fotometrijski*.** E4 je destilovao
  dva studenta iz učitelja koji se razlikuju samo po prozoru W: brightness
  retention na kraju rollouta 50.9% (W=2) vs 90.7% (W=20), što ogleda
  papirine 74% vs 101%. Implementacija me je naterala da primetim *zašto*
  maska Ω izbacuje prvi generisani chunk (učitelj tretira chunk 0 kao
  image prior; student kao nastavak — matchovanje tamo uči flicker) i
  zašto je detached-history backward nosiv (zajednički graf ionako ne bi
  bio dostupan).

## Šta me iznenadilo / bilo teže nego očekivano

- **Najiskrenija ablacija je u apendiksu.** Teacher–critic detectability se
  jedva menja posle W = 2 chunka kroz 13 poremećaja — dakle korist od drifta
  *ne* dolazi od sirove dužine prozora nego od celog istreniranog
  long-horizon učitelja plus rollout distribucije. Rad ovo kaže o sebi,
  što retko vidim.
- **VBench-Long rang 7 od 10, rečeno otvoreno.** Evoke pobeđuje na WBench i
  VBench-2.0 ali je sredina paketa na VBench-Long protiv many-step
  konkurencije — i rad to ističe umesto da sakrije, napominjući da
  poređenje nije step-matched. Stability claim (kosinus identiteta scene
  plato na 0.523 — tačno gde realni snimci ocene sebe 60 s apart) je
  uokviren kao *dokaz protiv degradacije*, ne kao fidelitet. Osvežavajuće.
- **Najteži toy deo: učiniti drift *identifikabilnim*.** Da bih pokazao
  da horizont važi u minijaturi, morao sam da drift bude apsolutan po
  poziciji u rolloutu (W=2 student bukvalno nikad ne vidi pozicije ≥ 2),
  inače kratki i dugi učitelji nauče istu korekciju. Papirini self-forced
  rollouti rade taj isti posao u pravom sistemu.
- **Studentska 3-korakna piramida je najmanje dokumentovan deo** (geometrija
  se ubrizgava samo na najgrubljem od 3 latentna nivoa, 12×20 → 48×80) i
  moj zamenski deo — conditioning net preko toy piramide — je najslabija
  karika moje reimplementacije. *Mehanizmi* (budžeti, banka, sparse
  attention, windowed distribution matching) su verni; generativni model je
  zamena. Obim iskreno zabeležen u `implementation/README.md`.

## Reference

- Rad: https://arxiv.org/abs/2608.13546
- Projektna stranica: https://evoke-world.github.io/Evoke/
- Moja implementacija: `implementation/` (pokreni `python3 train.py`, ~15 s
  CPU, piše `results.json`)
- Breakdown: `breakdown.md`
- Beleške sa čitanja: `notes.md`
