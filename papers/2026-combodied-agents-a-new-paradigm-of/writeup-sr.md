# Writeup — Combodied Agents: novi paradigma humane agentne AI

> Rad: Ding et al., 2026 · arXiv:2608.10915
> Moje viđenje nakon dva čitanja i implementacije od nule (toy verzija).

## Verzija u jednom pasusu

Skoro svi agenti koje danas gradimo deluju na *stanja koja nisu čovek*:
digitalni agent menja softverska stanja, embodiovani (roboti) fizička, i oba
se ocenjuju po uspešnosti zadatka. Ovaj rad predlaže treću klasu —
**Combodied Agents** (Companion + Body) — čije akcije ciljaju *samo stanje
osobe* (zdravlje, spoznaju, navike, sposobnost, odnose), kroz zatvorenu
petlju: percepcija po događajima → verovanje o latentnom stanju čoveka →
Personal World Model uslovljen akcijom → intervenciona politika ograničena
dopustivošću → povratna informacija u longitudinalnu memoriju. Radikalan
nije loop, nego tvrdnja ugrađena u evaluacioni kontrakt: **uspeh zadatka i
korist za čoveka se mogu razdvojiti, i kad se razdvoje, očuvanje agencije
(autonomija, sposobnost, dostojanstvo) mora biti nenadoknadivo** — nikad
nešto što se može "otkupiti" većom stopom završenih zadataka.

## Problem

Uzmite scenu iz uvoda rada: starija osoba preskoči dozu leka. Digitalni
agent može ponovo poslati notifikaciju. Robot može doneti kutiju sa lekovima.
Ni jedan sistem ne pita *zašto* je doza preskočena — zaboravio? zbunjen oko
uputstva? nuspojave? svesna odbijanja? — a pravi odgovor je potpuno drugačiji
u svakom slučaju: podsetnik pomaže kod `forgot`, besmislen je kod `confused`,
a aktivno razorno kod `refused`.

Niko to ne modeluje, jer ničiji *cilj* nije osoba. Asistenti drže profile
preferencija, companion chatbotovi optimizuju angažovanost (tačno suprotno
od potrebnog — nagrađeni su za zavisnost), zdravstveni wearables pale
alarme na populacionim pragovima bez pojma o saglasnosti ili
proporcionalnosti, a Human Digital Twins pokušavaju da replikuju čitavog
čoveka — neizvodljivo, neverifikabilno, i opasno po privatnost. F1 metrike u
međuvremenu *nagrađuju supstituciju po defaultu*: agent koji uradi sve
umesto korisnika dobija ocenu 10 dok korisnik gubi veštine.

## Ideja

Klasifikuj agente po **supstratu akcije** — klasi ciljnih stanja zbog kojih
akcije uopšte postoje. Taj jedan potez preusmerava sve:

1. **Definicija sa zubima**: pet zajednički neophodnih svojstava —
   modelovanje stanja čoveka, longitudinalnost, intervencija (ne samo
   odgovaranje), ko-agencija (adaptivna podela posla; zamena nije default),
   i očuvanje agencije. Chatbot koji zapamti ime ne prolazi test.
2. **Zatvorena petlja, formalizovana** (jed. 1–9): stanje čoveka `H_t` je
   latentno i posmatra se šumno; sledeće stanje zavisi od akcije agenta
   *i od akcije samog korisnika i egzogenih faktora* — baš zato su
   intervencije inherentno neizvesne i PWM mora da daje *kalibrisane
   distribucije* ishoda po alternativnim akcijama, ne tačkaste procene.
3. **Politika sa tvrdim podom**: prvo filter dopustivosti (saglasnost,
   bezbednost, opseg, neizvesnost, reverzibilnost — nenadoknadivi), pa tek
   onda Pareto argmax nad *vektorskom* korisnošću (zadatak, autonomija,
   sposobnost, odnos, dostojanstvo). Bezbednost i saglasnost ne mogu biti
   nadglasani korisnošću.
4. **Memorija koja čuva dokaze**: provenijencija, neizvesnost i — komponenta
   koja mi je delovala najnovije — **intervencio-response memorija** kao
   first-class evidencija. "Šta se desilo prošli put kad smo kod ove osobe
   probali X" je najvredniji i najprivatniji podatak sistema, što se direktno
   uvezuje u njihov edge-governance argument.

## Kako radi (intuicija)

Zamislite kliničarski način razmišljanja, formalizovan. Dobra medicinska
sestra ne maksimizira "doza popijena danas". Ona drži diferencijalnu
dijagnozu (verovanje `Z_t` o tome zašto je doza preskočena), mentalni model
odgovora *ovog* pacijenta (PWM) i jak osećaj šta sme a šta ne sme bez
saglasnosti (skup dopustivosti). Ponekad je prava akcija **ćutanje**, jer je
poverenje imovina koja se ne vidi ni u jednoj metrici jednog dana, a koja
određuje uspešnost svake buduće intervencije.

Duboka projektantska odluka petlje: **agent kontroliše samo jedan ulaz u
tranzicionu funkciju čoveka.** `H_{t+1} ~ T_H(H_t, a_agent, a_user, Ξ)`.
Korisnik može odbiti, navicirovati se, ili se uvrediti. Zato rad inzistira da
PWM bude *kalibrisan* — tačkasta procena "podsetnik → doza popijena" je
upravo ona lažna sigurnost koja uništava odnose na veliko.

## Šta sam naučio implementirajući

(Iz `implementation/` — toy medication-companion: `PersonSim` sa tri latentna
uzroka preskakanja, istrenirani `BeliefNet`, `PWMEnsemble`,
`InterventionMemory`, i tri politike upoređene kroz 120 dana × 30 osoba.)

- **Razdvajanje zadatka i koristi se trivijalno reprodukuje — i to je poenta.**
  Moja naivna politika (argmax P(take)) je postigla 0.907 adherence... tako
  što je u 100% slučajeva eskalirala negovatelju, srušivši kvalitet odnosa sa
  1.00 na 0.03 i samo-sposobnost na 0.07. To patološko ponašanje nigde nisam
  hard-kodirao; ono je *ispalo* iz optimizacije očigledne metrike. Centralna
  briga rada nije hipotetička — ona je default ponašanje agenta orijentisanog
  na uspeh zadatka.
- **Filter dopustivosti odradi celu "proporcionalnost" bez ijednog
  parametra koji se uči**: četiri tvrda ograničenja, pa `U = P_take +
  5·Δautonomija + 4·Δodnos`. Combodied politika je izabrala *ćutanje* 13 puta
  po osobi, podsetila 12, coachovala 19, eskalirala 3 — i kupila 10× veći
  gain u adherence-u u odnosu na "bez agenta", držeći odnos/samo-sposobnost
  na ≈ 0.94 vs 0.03/0.07 kod naivne. Sve rade težine nad deltatima agencije,
  ne pamet.
- **Intervencio-response memorija je ono što ga čini ličnim.** Gledati
  take-rate jedne osobe kako ide 0.33 → 1.00 kako se lični ishodi
  akumuliraju — sa ponovljenim podsetnicima koji navikavaju (verovatnoća
  uzimanja opada sa brojem ponavljanja) — nateralo me je da shvatim da ova
  komponenta radi ono što Bayesian optimizacija radi za hiperparametre, samo
  za *površinu odgovora čoveka*.
- **Kalibracija PWM-a je merljiva i netrivijalna čak i u toy verziji.** Moj
  ansambl je predvideo remind na 0.62 vs ostvareno 0.62 (odlično), ali
  clarify na 0.70 vs ostvareno 0.89 — retki uzroci su sistematski
  potcenjeni. Skalirajte to na prave ljude i jasno je zašto rad tretira
  nekalibrisan PWM kao bezbednosni problem, a ne problem tačnosti.

## Šta me iznenadilo / šta je bilo teže nego što sam mislio

- **Ista akcija, suprotan efekat po latentnom stanju — emergiralo iz
  treninga bez ikakvog kodiranja**: `remind` je naučio 0.67 za `forgot` ali
  0.11 za `refused`, `coach` 0.51 za `refused` a 0.39 za `forgot`. Očekivao
  sam da ću morati da tu strukturu ubacim ručno; uslovni trening podaci su
  je uradili sami.
- **Najteža projektantska odluka bile su težine korisnosti** (5 za
  autonomiju, 4 za odnos, 1 za zadatak). Pareto okvir rada izbegava fiksiranje
  brojeva, ali svaki runnable sistem mora da bira — a ti brojevi *jesu*
  etika. Ovo me je ubedilo da klauzula "user-approved selection rule" u radu
  je noseća, ne boilerplate.
- **Position paper je zaista implementabilan** — ali samo jer se obavezao na
  jednačine. Da je ostao na nivou taksonomije, ne bi bilo šta da se gradi.
  Distinkcija kontrakt-recept: ovaj rad je kontrakt, a moj toy je jedan
  potpis na njemu.
- Šta je bilo *lakše* nego što sam mislio: belief net postiže 99.9%
  tačnosti uzroka na toy karakteristikama. Inferencija latentnog stanja nije
  usko grlo; tranzicioni model i politika su gde je sva prava težina (i sav
  pravi rizik).

## Reference

- Rad: https://arxiv.org/abs/2608.10915
- Moja implementacija: `implementation/` (pokrenite `python3 train.py`, ~7 s CPU)
- Breakdown: `breakdown.md`
- Beleške iz čitanja: `notes.md`
