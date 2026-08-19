# Writeup — MerchantBench: benchmarking LLM agenata za dugoročnu koherenciju u e-commerce operacijama

> **Paper:** MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations
> **Autori:** Qiming Shi, Yulong Tao, Linbo Jin, et al. (Alibaba Group, ZJU, PKU, Fudan)
> **Godina:** 2026 · **ArXiv:** https://arxiv.org/abs/2607.28956

Ovo je moje vlastito objašnjenje, napisano nakon čitanja i reimplementacije
rada. Sinteza, ne prepričavanje apstrakta.

## Verzija u jednom pasusu

MerchantBench nije nov model — to je lenjir. Autori tvrde da ne znamo
mjeriti **dugoročnu koherenciju (Long-Term Coherence, LTC)**: sposobnost
agenta da ostane svrsishodan kroz stotine odlučnih tačaka, prilagođavajući
se evidenciji koja stiže s *različitim kašnjenjima*. Njihov odgovor je
simulacija veleprodajne prodavnice na platformi 1688 (Alibabaov B2B
market), 365 dana, sat po sat, utemeljena na 98.843 stvarna produkta, u
kojoj LLM *biva* trgovac: svakih 12 sati dobija 26 alata za nabavku,
upravljanje sa 50 listing-slotova i čuvanje keša — pri čemu *narudžba*
obavezuje novac danas, a njen ishod (refundacija, loša recenzija) ispliva
tek nekoliko dana kasnije. Kad se 8 frontier LLM-ova pusti kroz tu godinu,
najbolji završava sa 27,3% imovine koju skupe tri ljudska igrača.
Najzanimljivije je *kako* padaju: ne biranjem loših produkata, nego
postepenim utihnjivanjem — agenti sve manje djeluju, krivo pamte krajnji
datum, ili se skvrče na reaktivno gašenje požara sa dobavljačima dok im
portfelj trune.

## Problem

Benchmark-i za agente mahom testiraju *ograničene* zadatke: odgovori na
pitanje, kupovina na webu, navigacija kroz sobu. Uspjeh je binaran i
neposredan. Prava produkcija nije takva. Deployirani agent se suočava sa:

1. **Dugim horizontom** — stotine do hiljade odlučnih tačaka, nagrada se
   materializuje tek na kraju.
2. **Akcijama koje ograničavaju budućnost** — novac danas vezan nabavkom
   je novac koji sutra ne možeš potrošiti; slot zauzet dud-produktom
   blokira dobar produkt.
3. **Feedback-om miješanog kašnjenja** — promjene cijena dobavljača vide
   se u roku sata; da li će narudžba završiti refundacijom ili
   jednom zvjezdicom saznaje se danima nakon odluke koja ju je uzrokovala.
4. **Kumulativnim, mjerljivim posljedicama** — nekherentnost se sabire u
   završni bilans.

Nijedan postojeći benchmark ne kombinuje sve četiri. Zato model koji
odlično prolazi WebArena može godinu dana driftovati u nekherentnost i to
neko ne primijeti do deploya.

## Ideja

Uzeti prodavčku stranu e-commercea kao LTC poligon i napraviti **neto
imovinu nakon 365 simuliranih dana** jedinim, nefakeabilnim skorom.
Formalno, konačno-horizontni POMDP (satni taktovi, 730 odlučnih prozora,
nula međunagrade, J(π) = E[konačna neto imovina] = keš + depozit +
novac u tranzitu + potraživanja). Dizajnerske odluke koje to nose:

- **Simulacija na nivou narudžbe sa drop-shippingom.** Tražnja stiže kao
  Poisson događaji po listingu: λ = D·w·r·ℓ·(p/p_ref)^{−ε} — stvarna
  dnevna tražnja × satni profil × multiplikator ocjene prodavnice × rampa
  izloženosti listinga × cjenovna elastičnost. Svaka narudžba odmah
  nabavlja od dobavljača (tereti keš), pa prolazi lifecycle: Placed →
  Procured → Shipped → Delivered → Settled, moguće sa otkazom,
  refundacijom ili lošom recenzijom — pri čemu je ishod svake narudžbe
  **presampleovan pri rođenju**, a otkriva se tek pri dostavi.
- **Asimetrični feedback.** Uzvodni događaji dobavljača (poskupljenja,
  delistiranja, kašnjenja) udaraju odmah; nizvodni ishodi narudžbi stižu
  danima kasnije i moraju se pratiti nazad do odgovorne odluke. Agent
  mora voditi svoje knjige.
- **Jedna zajednička ocjena kažnjava lijenost.** Svaka abnormalna
  narudžba vuče ocjenu prodavnice niz, a ocjena multiplikuje *svu*
  tražnju (0,10×–1,20× po zvjezdanim opsezima). Jedna loša produktna
  familija može izgladniti cijeli portfelj.
- **Uslov smrti.** Kazne idu sa sigurnosnog depozita; depozit na nuli —
  prodavnica se trajno zatvara. Nekherentnost nije samo suboptimalna,
  može biti terminalna.

## Kako radi (intuicija)

Zamisli traku za trčanje sa odloženim bolom. Agent zarađuje maržu
(≈21 RMB po narudžbi na skali rada) samo ako drži 50 slotova punim
sezonski relevantnih produkata po cijenama koje tržište prihvata.
Tražnja u katalogu je *nestacionarna* (vrhovi 618 i 11.11 promocija,
dno za kineske Nova godina), pa prošlomjesečni pobjednici jenjavaju i
agent mora stalno iznova nabavljati. U međuvremenu, svaki prozor u
kojem ne djeluje se kumulira: prazni slotovi ne zarađuju, propušteni
rokovi koštaju kazne, stari listingi gube rampu izloženosti.

Analitički doprinos rada je dekompozicija LTC-a na:

- **Operacionu koherenciju** — da li nastavljaš *djelovati*? Mjeri je
  **Sustained Working Rate (SWR)**: minimum, po svim kotrljajućim
  30-dnevnim prozorima, udjela odlučnih prozora sa ≥1 pozivom alata.
  Ljudi: 100%. LLM-ovi: 10,6–99,4%.
- **Stratešku koherenciju** — da li i dalje djeluješ *ka cilju, uz
  evidenciju*? Imenovani modusi pada: *Control-Loop Narrowing* (sourcing
  petlja se skvrči na reaktivno gašenje požara), *Premature
  Abandonment* (Kimi K2.6 run je na dan 104 zaključio da se prodavnica
  ne može oporaviti i ušutio 355 od 523 preostala prozora), greške
  memorije (run je na dan 282 krivo zapamtio dan 285 kao kraj i prestao
  puniti slotove), i ravne trajektorije nabavke tamo gdje ljudi eskaliraju
  cijene (43→91 RMB) kako likvidnost raste.

Dijagnostika koju dodaju: **Time-aware Sourcing Gain** — poredi
usklađenost tražnje stvarnog mjesečnog miksa listinga sa kontrafaktičkim
scenarijem da je januarski miks ostao fiksiran cijele godine. Pozitivan
dobitak = prava sezonska realokacija, i korelira sa konačnom imovinom.

## Šta sam naučio implementirajući

(Stvari koje su se razjasnile tek kad sam napisao kod — `implementation/`,
čisti Python, bez zavisnosti.)

1. **Rampa izloženosti ℓ je skriveni kurikulum.** Novi listing kreće na
   20% izloženosti i rampa do pune kroz 14 dana, pa opada (κ=0,0092,
   pod 0,10). Ta jedna konstanta nameće dugoročno ponašanje: miopni
   agent vidi „novi listing = slaba prodaja" i delista, uništavajući
   upravo ono što bi se isplatilo. Pola razlike između mog random
   trgovca (65k) i rule trgovca (205k) je samo *ostavljanje dobrih
   listinga na miru*.
2. **Presampleovani ishodi čine ekonomiju poštenom.** Pošto je sudbina
   svake narudžbe fiksirana pri rođenju, okolina ne može slučajno
   nagraditi sreću; jedina poluga agenta je *koje rizične profile
   pušta kroz vrata*. U mom sintetskom katalogu, biranje pogrešnog
   repa raspodjele rizika košta ~40% imovine kroz kanal ocjene —
   sprega (loše narudžbe → ocjena → tražnja svih listinga) je mnogo
   čvršća nego što proza rada sugeriše.
3. **Likvidnost je pravi limit rano, ocjena kasnije.** Sa 2.000 RMB
   početnog keša i trenutnim nabavnim terećenjima, prvi mjesec je
   cash-flow zagonetka (moj rule trgovac budžetira dodavanje listinga
   protiv potraživanja koja se mogu poravnati). Do trećeg mjeseca keša
   ima, a vezujući ograničeni faktor je popravka ocjene. Problem
   *mijenja oblik* tokom horizonta — to statični benchmark-i ne mogu
   naučiti.
4. **SWR je trivijalno gameable, a ipak dijagnostički.** Sat koji u
   svakom prozoru pozove jedan alat ima SWR=100%. Metrika nešto znači
   samo u paketu sa neto imovinom — „jesi li *bio živ* I solventan".
   Rad to razumije; čitanje samo SWR tabele bi obmanulo.
5. **Kalibracija na skalu rada je finoća.** Ljudi su prosječno imali
   9.442 narudžbe godišnje (~26 dnevno) i 217,6k imovine. Dovesti
   sintetski katalog od 200 produkata do tih redova veličine (≈21 RMB
   marže po narudžbi) trajalo je koliko i sam lifecycle FSM. Ekonomija
   po narudžbi u radu tiho radi puno posla.

## Šta me iznenadilo / bilo teže nego što sam očekivao

- **Koliko loše frontier modeli prolaze naspram glupe skripte.** Rule
  baseline iz samog rada ima 24,48k — a *većina od 48 LLM runova* ga
  jedva prelazi, najbolji je na 59,46k naspram ljudskih 217,61k. Moj
  prilično orakulski rule trgovac na mojoj sintetskoj ekonomiji došao je
  do 205k, tj. ~94% ljudskog prosjeka — pa jaz nije u tome da je zadatak
  *težak* po sebi; nego LLM-ovi padaju na *održanom, samousmjerenom
  operiranju* i kad je svaka pojedinačna odluka laka.
- **Priče o padu čitaju se kao organizacione patologije**, a ne
  ograničenja modela: prerano odustajanje od izlječive prodavnice,
  kargo-kult delistanje („manje listinga koncentriše saobraćaj" — jedan
  Claude Opus 4.8 run se na toj teoriji skvrčio sa 47 na 3 listinga),
  krivo zapamćeni rokovi. To su greške *održavanja uvjerenja*, što
  LTC preusmjerava djelimično na problem memorijskih sistema.
- **Hermes-style framework-i pobjeđuju ReAct na 7/8 modela** (+53,3%
  prosječne imovine) — scaffolding koji upravlja kontekstom i
  vještinama značio je više od izbora modela. Za benchmark rad, to je
  snažna systems-lekcija skrivena u dodatku.
- Najtrickiji kod bio je prozaičan: **satovi poravnanja vs. ishoda**
  (novac se poravnava ≤168h nakon dostave, ali posljedica po ocjenu/kaznu
  može opaliti na *drugom* satu). Dekuplirati ta dva bez duplog
  brojanja kazni trajalo je tri pokušaja.

## Reference

- Paper: https://arxiv.org/abs/2607.28956
- Zvanični kod: https://github.com/KhanCold/merchantbench
- Moja implementacija: `implementation/` (pokretanje: `python3 train.py`,
  ~2s, čisti stdlib; rezultati u `implementation/results.txt`)
- Breakdown: `breakdown.md` · Bilješke iz čitanja: `notes.md`
