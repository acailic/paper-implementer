# Writeup (srpski) — Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA

> Moje sopstveno objašnjenje rada, kao da ga prepričavam kolegi koji ga nije
> pročitao. Ovo **nije** rezime apstrakta — nego sinteza nakon čitanja i
> implementacije.

**Rad:** "Macaron-V1: Towards Open Continual Learning with Self-Improvement
and Mixture-of-LoRA" — Mind Lab, 2026, arXiv:2608.09819.

## Verzija u jednom pasusu

Macaron-V1 je sistemski izveštaj (ne klasičan metodološki rad) koji se kladio
na jednu veliku ideju: **zamrzni bazni model zauvek, svaku sposobnost dodaj
kao zaseban LoRA adapter, svaki potez korisnika rutiraj tačno na jedan
adapter, i tretiraj harness (alate, prompte, runtime konfiguraciju) kao
first-class, verzionisan cilj treninga koji sam model može da prepiše u
jezičkom prostoru.** Ono što se trenira nije samo težine nego *par
model–harness*, poboljšavan rekurzivnom petljom discover → expand → update.
Flagship "Venti" je zamrznut GLM-5.2 od 744B plus četiri LoRA od po ~7.7B
vrednosti (chat, agent, coding, GenUI), pri čemu chat adapter ujedno služi i
kao router kroz constrained decoding.

## Problem

Post-trening danas je centralizovan i statičan: podešiš model na snapshot
zadataka, objaviš checkpoint, i on se nikad više ne poboljšava iz produkcijskog
iskustva. Iz toga prate dva moda otkaza:

1. **Interferencija između zadataka.** Zajednički post-trening nad
   heterogenim zadacima (chat + tool-use + coding + UI generacija) čini da
   različito oblikovani chain-of-thought konkurišu za deljene parametre.
   (Rad je iskren da poređenje sa budžetno-izjednačenim jednim LoRA adapterom
   *nije* u ovoj verziji — interferencija je motivacija dizajna, ne izmerena
   tvrdnja.)
2. **Svet ide dalje.** Novi alati, znanje i potrebe korisnika stižu posle
   objave; zamrznut checkpoint aproksimira statični optimum.

Cilj je "experiential intelligence": **adaptacija** (rekurzivno poboljšanje
verzionisanih parova model–harness) i **kolaboracija** (kompozicija
posebno treniranih specijalista).

## Ideja

Tri spojena poteza:

1. **Mixture-of-LoRA kao orkestracija, ne spajanje.** Adapteri se nikad ne
   merge-uju ni guraju jedan na drugi; kompozicija se dešava na nivou
   zahteva kroz rutiranje. Svaki odgovor se pripisuje tačno jednom
   specijalisti, a nova sposobnost je nov (A, B) par na netaknutoj bazi.
2. **L0 je router.** Nema zasebnog router modela, nema keyword pravila: chat
   adapter klasifikuje svaki potez u jednu od četiri kanonske labele uz
   budžet od 24 tokena i constrained-decoding gramatiku (samo su 4 legalne
   labele uopšte dekodabilne). Zahtev se prenosi kao citirani *untrusted
   text* — elegantna odbrana od prompt injection-a.
3. **Harness je trenabilan u jezičkom prostoru.** HCP, verzionisani TOML
   ugovor (alati, prompti, veštine, hooks, env), ne nosi gradijente — ali
   model može da ga *prepiše*, predloži izmene, ponovo pokrene pogođeni
   slice i objavi preživele kao sledeću konfiguraciju. Ključni dokaz: na 122
   zadatka u kojima zamrznuta baza *ne uspeva ni u jednom*, sama adaptivna
   pretraga konfiguracije (bez ikakve promene težina!) diže pokrivenost sa
   2/122 na **122/122**. Neuspeh pod baznom konfiguracijom ne znači da
   sposobnost nedostaje.

## Kako radi (intuicija)

Serving petlja po potezu korisnika je **Route → Answer → Summary**:

- **Route**: L0 dekodira jednu labelu (constrained gramatika, ~0.5 s).
- **Answer**: izabrani specijalista odgovara iz svog **own-view**-a — svoji
  raniji potezi *verbatim* (ceo tool trace), potezi svih drugih specijalista
  sažeti u ≤192 tokena.
- **Summary**: specijalista koji je odgovorio emituje ≤192-token sažetak,
  koji se čuva samo na serveru (nikad se ne pokazuje klijentu).

Own-view je tiho briljantan deo. Pošto se kontekst svakog adaptera
*deterministički rekonstruiše* iz append-only timeline-a — sopstveni potezi
bajt-identično, tuđi uvek sažeti — ponovni ulazak u LoRA daje bajt-identično
proširenje prefiksa prethodne posete. Nativni prefix cache engine-a jednostavno
pogodi. **Ponovna upotreba KV-a po adapteru je emergentno svojstvo stabilnog
formata prompta, ne feature engine-a.**

Okolo modela sede: stateful **REPL agent harness** (zavisne vrednosti
ostaju kao promenljive — case study radi u 6 poteza ono što traži 48
poteza sa jednim pozivom po potezu; samostvoreni alati se promovišu tek
posle privatne validacije), **UI4A** (agent piše običan frontend kod unutar
runtime-omogućenih granica; svaki gest korisnika je strukturirani Action sa
NoAI vidljivošću), **MindForge** (kontrolna ravan RSI: problem banke →
evaluirane trajektorije → datasetovi → sledeći model + sledeći HCP, sve
povezano lineage-om), **MinT** (životni ciklus adaptera: adapter-only handoff
18.3× brži od merged checkpointa) i **LongStraw** (response-only trening za
dugačke kontekste: forward stanje prompta uhvati jednom bez autograda, pa
replayuj po jedan odgovor sa autogradom — vršna memorija drži samo graf
najdužeg jednog odgovora).

## Šta sam naučio implementacijom

(Iz toy re-implementacije u `implementation/` — zamrznuta mala baza, 4 LoRA
slota, prava Proxy petlja, sve na CPU.)

1. **Interferencija između zadataka je stvarna i vidljiva čak i u toy
   razmeri.** Ablacija koja u radu nedostaje — budžetno izjednačen jedan debeli
   LoRA protiv MoL — bilo je prvo što sam pustio. MoL prosek 0.750 protiv
   fat 0.494, i mod otkaza je tačno motivacija rada: debeli adapter sruši GUI
   specijalistu na 0.00 dok svaki namenski slot drži svoju veštinu na 1.00.
   Ništa u radu to *ne meri*; u toy razmeri je očigledno.
2. **Ponovna upotreba KV prefiksa je disciplina formatiranja, ne feature
   engine-a.** Renderovanjem svakog sopstvenog poteza bajt-identično zahtevu
   koji ga je proizveo, uzastopni own-viewovi se savršeno ugnježđuju
   (stabilnost prefiksa 1.000; ~58% konteksta svakog zahteva je reusable
   prefiks u 6-poteznim mešovitim konverzacijama). "Trik" ne košta ništa u
   treningu — čisto je stvar toga kako Proxy rekonstruiše liste poruka.
3. **Rutiranje kroz constrained decoding je trivijalno pouzdano.** Routing
   accuracy 1.000 (rad: 0.9912) kad je prostor labela 4 a gramatika tvrda.
   Tvrdo ograničen decode pretvara router u problem formatiranja, a ne
   klasifikacije.
4. **Sekvencijalni trening katastrofalno zaboravlja — joint mixture trening
   popravlja.** Trening L0 chat-a, pa njegove router dužnosti, uništio je
   raniju veštinu (generacija se raspala u gibberish). Mešanje dužnosti u
   jednu zajedničku SFT distribuciju vratilo je obe. To je pravilo
   klasterovanja iz rada ("grupiši zadatke koji dele obrazac razmišljanja u
   jedan LoRA") koje se pokazuje kao tvrdo ograničenje, ne preferencija.
5. **Pitanje 192-token sažetka ostaje otvoreno u toy razmeri.** Summary-handoff
   protiv pune istorije nije pokazao merljivu razliku (0.333 vs 0.333) — u
   ovoj razmeri ekstraktivni sažetak čuva ključne tokene, a *echo format* je
   slabija karika. Otvoreno pitanje iz rada je stvarno otvoreno.

## Šta me iznenadilo / bilo teže nego što sam očekivao

- **Iskrena evidencija me iznenadila (pozitivno).** Izveštaj više puta
  sam ograđuje sopstvene tvrdnje: routing accuracy izmerena na trening
  podacima zove se dijagnostika, ne generalizacija; tabela zadržavanja
  multimodalnosti nosi oznaku "not established preservation"; rezultat
  122/122 je eksplicitno plafon pokrivenosti, ne generalizacija na
  nepoznate zadatke. Sistemski izveštaji obično nisu toliko disciplinovani.
- **Char-level aritmetika sa više cifara nije bila naučiva** modelom od 4
  sloja sa rank-8 LoRA u mom budžetu — L1 exact-match je ostao 0.00 (SFT loss
  se zaustavi ≈0.13: format naučen, cifre nepouzdane). Prijavljujem takvo
  kakvo jesti; zanimljivo je da fat-LoRA ruka pada isto (0.05), pa poređenje
  ostaje pošteno.
- **Koliko malo rada su uopšte težine.** Posle prvog čitanja misliš "4 LoRA-e
  na zamrznutom divu". Posle drugog shvatiš da su nosivi artefekti format
  timeline-a, HCP ugovor i promocijske kapije — stvari koje nikad ne prime
  gradijent.

## Reference

- Rad: https://arxiv.org/abs/2608.09819
- Moja implementacija: `implementation/` (pokreni `python3 train.py`, ~6.5 min
  CPU; stvarni izlaz u `implementation/run_output.txt`)
- Breakdown: `breakdown.md`
- Bilješke iz čitanja: `notes.md`
