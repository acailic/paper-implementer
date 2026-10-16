# Writeup — Demystifying Agent Skills: Why They Work—Until They Don't (srpski)

> **Rad:** Demystifying Agent Skills: Why They Work—Until They Don't
> **Autori:** Zhiyuan Jiang, Fangrui Huang, Hanwen Xing, Xander Wu, Yipeng Gao,
> Rui Cao, Mengdi Wang, Shilong Liu, Yijiang Li
> **Godina:** 2026 · **arXiv:** [2608.14036](https://arxiv.org/abs/2608.14036)

## Verzija u jednom pasusu

Skillovi (kompaktni `SKILL.md` proceduralni paketi koje agent pronalazi u svom
okruženju) rade — ali ne iz razloga koje većina ljudi pretpostavlja. Kroz
kontrolisane, uparene eksperimente autori pokazuju da skillovi pomažu zato što
deluju kao **proceduralna sidra**: stabilizuju *izvršavanje* (redosled
podešavanja, sekvence alata, navike verifikacije, izbegavanje zamki), a ne zato
što ubrizgavaju nedostajuće *činjenice* (65,7% slučajeva gde je skill presudio
pripisano je sidrenju, naspram svega 4,5% ubrizgavanju znanja). Dalje, ceo
proces korišćenja skillova se rastavlja na nezavisna uska grla — reprezentacija,
anotacija ishoda, transfer, pretraga — i svako se lomi na drugačiji način:
destilovani skill pobeđuje sirovi workflow memory za +6 poena u uparenim
uslovima, destilacija je ono što artefakt čini prenosivim između agentskih
frejmworka, a preciznost pretrage kolabira (29,6% → 3,3%) kad bazraste na 100
skillova, dok krajnji uspeh *ostaje ravan* — savršena pretraga nije ni
potrebna ni dovoljna.

## Problem

Svi danas kače "skillove" na LLM agente, i svi ih evaluiraju na isti pokvaren
način: pokreni agenta sa i bez skilla, uporedi agregatne stope uspeha. Taj jedan
broj ne može da ti kaže:

- da li je pomogao **sadržaj** skilla ili samo njegov **format/pozicija**;
- da li je neuspeh došao od **pogrešno pretraženog skilla** ili od **loše
  primenjenog pravog**;
- da li bi isti artefakt preživeo **drugi agentski harness**;
- u šta da investiraš kad gradiš self-evolving agenta (bolji destiler? bolji
  retriver? bolje labele?).

Bez odgovora na nivou mehanizma, biblioteke skillova su kargo kult.

## Ideja

Pokreni kontrolisane eksperimente gde je **sve osim jednog faktora
fiksirano**, a onda labeliraj *zašto* je svaki trial uspeo ili pao taksonomijom
koja je **izvedena iz podataka** (open coding), a ne pretpostavljena. Ključne
kontrole:

1. **Isto izvorno iskustvo, drugačija reprezentacija.** Iz jednog pool-a
   trajektorija sa fiksnom mešavinom uspeha/neuspeha (5s0f … 0s5f) napravi i
   Workflow Memory (očišćeni tragovi, ubačeni inline u prompt) i Skill
   (destilovani SKILL.md, smešten u okruženje). Svaki uporedi sa sirovim
   agentom.
2. **Ablacija anotacije ishoda.** Rebuilduj skillove bez success/failure labela
   u inputu destilera. Ako dobit preživi, radi sadržaj — ne labele.
3. **Transfer između frejmworka.** Artefakti destilovani iz Codex runova,
   izvršeni unutar Gemini CLI-ja.
4. **Dekompozicija pretrage.** Bazeni skillova veličine 5→100 sa distraktorima
   koji su random, semantički slični ili neslični — mereno na tri
   nezavisna nivoa: offline embedding top-1, eksplicitni odabir agenta, i
   stvarna upotreba tokom izvršavanja.

Zatim dva kruga open codinga nad stratifikovanim transkriptima daju
**taksonomiju od 12 modova u 3 super-kategorije** (uspeh / execution neuspeh /
invocation-boundary neuspeh), validiranu na 95,8% saglasnosti sa ljudima
(κ = 0,952), a svi upareni kontrasti dobijaju bootstrap CI od 1.000 resampleova.

## Kako radi (intuicija)

Najčistiji mentalni model: masa neuspeha agenta je *očuvana, ali
redistributivna*. Sirov agent pada uglavnom na execution stvarima — podešavanju
okruženja, šemama izlaza, zaboravljanju runtime verifikacije. Daj mu bilo koji
proceduralni artefakt i ti neuspesi se prepolove; agent sada uspeva "zato što
mu je artefakt rekao" (`skill_guided_success` skače na 61,6% dok
`autonomous_clean_success` kolabira na ~0 — strukturalne nule koje imaju smisla
čim ih vidiš). Ali artefakt unosi i *svoje* modove neuspeha: workflow memory
davi agenta u šumu procesa (timeouti 10,6% vs 1,7% raw), a skillovi dodaju novu
kategoriju — `skill_guidance_misapplied_or_ignored` (10,0%) — agent otvori
pravi skill i upropasti ga ili ignoriše.

Zašto destilacija pobeđuje memoriju? Workflow memory čuva *doslovni oblik*
svojih izvornih tragova, sa svim mrtvim granama. Skill enkoduje *pattern*. Radni
primer iz rada (A.3) je savršen: na React performance tasku, sirovi agent
preterano sa keširanjem i pada na API-call testu; WM prolazi taj test ali zadržava
serijalizovanu strukturu fetcha i pada na latenciji (915 ms > 800 ms); skill kaže
"pokreni promise rano, await kasno, paralelizuj nezavisne fetchove" — prolazi
svih 11 testova na 711 ms. Isto izvorno iskustvo; reprezentacija odlučuje da li
nasleđuješ slučajnost ili princip.

Zašto pretraga jedva da išta menja? Zato što skill koji je *približno* tačan i
dalje sidri proceduru. Agent over-selektuje iz velikih bazena (preciznost u
podrum) dok uspeh ostaje ravan — efekat sidrenja je robustan na biranje
skilla koji je blizu, barem dok distraktori nisu advezarijalno slični.

## Šta sam naučio implementirajući

Moja re-implementacija (`implementation/`, Monte-Carlo simulacija celog
eksperimentalnog aparata u čistom stdlib Pythonu — ruke, mešavine, destilacija,
transfer, pretraga, plus agregaciona matematika sa uparenim bootstrap CI-jevima)
učinila je nekoliko stvari konkretnim koje rad ostavlja implicitnim:

- **Upareni dizajn je cela poenta.** Težina taskova divlje varira; čim
  računaš Skill−WM delte *unutar* task+setting trojki i bootstrapuješ preko
  trojki, nivo šuma padne toliko da je +6 poena efekta čisto signifikantan. U
  mom sintetičkom svetu ista mašinerija je dala +8,50 poena, CI [+5,56, +11,67],
  iz efekata koje bih potopio da sam poredio neuparene agregatne stope.
- **Strukturalne nule proizlaze iz logike labeliranja, ne iz fizike.** Čim
  artefakt postoji i trial uspe, sudija pripisuje uspeh artefaktu — "autonomni
  uspeh sa učitanim skillom" je gotovo neoservljiv *po konstrukciji*. Ovo me je
  nateralo da mnogo pažljivije čitam Tabelu 11: te ćelije od 0,4/0,2% su koliko
  mereni artefakt, toliko i bihejvioralna činjenica.
- **Definicije preciznosti/recall-a odrađuju pravi posao.** Brojanje svakog
  distinct dodirnutog skilla jednom, P=0 za prazne predikcije, i F1 iz
  agregiranih P/R (ne prosek po rekordu) — bilo koja greška ovde menja RQ4
  priču suštinski. Jednačine 1–2 su kratke; nose krov.
- **Grid mešavina je tiho najbolja ablacija u radu.** Monotona degradacija od
  5s0f do 0s5f, pri čemu skill degradira gracioznije od WM-a, jeste ono što me
  je ubedilo da je priča o sidrenju uzročna, a ne samo "više konteksta pomaže".

## Šta me iznenadilo / bilo teže nego što sam očekivao

- **Workflow memory može biti gori od ničega** — nošenje mrtvih grana inline
  aktivno šteti (timeouti ×6 vs raw). Očekivao sam "WM < Skill"; nisam
  očekivao da je "WM < raw" živa mogućnost.
- **Kolaps preciznosti pretrage bez kolapsa uspeha** je duboko
  kontraintuitivno ako dolaziš iz RAG sveta, gde se kvalitet pretrage tretira
  kao plafon. Ovde je plafon proceduralna relevantnost, i to je mekan, fuzzy
  plafon.
- **Random-pool kuriozitet:** na random distraktorima, uspeh Codexa *raste* sa
  veličinom bazena (24,3 → 44,8) — ne-zlatni skillovi su ponekad stvarno
  korisni. Moja simulacija je ovo reprodukovala i to je nalaz za koji bih
  najviše voleo nastavak rada.
- Najteže za verno implementirati bila je **tro-nivovska dekompozicija u
  RQ4** (embedding / selekcija / izvršavanje ne smeju curiti jedni u druge —
  nijedna preselekcija ne prolazi do izvršavanja). Lako je slučajno učiniti
  Arm 3 "pravednijim" i time uništiti poentu rada.

## Reference

- Rad: https://arxiv.org/abs/2608.14036
- Zvanični kod: https://github.com/zhiyuanjiang04/demystify-agent-skills
- Moja implementacija: `implementation/` (pokreni `python3 train.py`, ~5 s, samo stdlib)
- Breakdown: `breakdown.md`
- Beleške iz prve/druge provale: `notes.md`
