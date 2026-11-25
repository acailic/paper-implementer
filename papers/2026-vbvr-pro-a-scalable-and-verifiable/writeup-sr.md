# Writeup (SR) — VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning

> Moje objašnjenje, napisano nakon dva pažljiva čitanja, pune raščlambe i
> from-scratch mini implementacije (**MiniVBVR**: 8×8 grid-world + flow-matching
> MLP + GRPO). Nije rezime apstrakta — ovo je šta zaista mislim da se dešava,
> plus šta me naučio kod.
>
> **Rad:** Junxiang Xu et al., *VBVR-Pro: A Scalable and Verifiable Suite for
> Native Visual Reasoning*, arXiv:2608.26105 (2026).
> Projekat: https://video-reason.com/ · Kod: https://github.com/Video-Reason/VBVR-Pro

## Verzija u jednom pasusu

„Nativno vizuelno rezonovanje" znači rešavati probleme *generisanjem* vizuelnih
stanja — nacrtaj rutu kroz lavirint, zaokruži objekte, dopuni obrazac — umesto
da izbaciš tekst o pikselima. Oblast nije mogla ozbiljno da ovo trenira jer
nije imala skalabilnu ponudu zadataka, pouzdanu nagradu ni način da poredi
video naspram slike naspram interleaved generisanja na ravnopravnoj osnovi.
VBVR-Pro zatvara sva tri odjednom: **300 proceduralnih generatora zadataka**
(ne 300 instanci — jedan generator, neograničeno seeded instanci, svaka sa
programskim solverom i metapodacima), **deterministički scoreri na nivou
atributa** (ocenjuju *svojstva* generisanih frejmova — pozicije, brojeve,
boje, kontinuitet trajektorije — nikad piksele) i **jedan te isti problem
renderovan u video / sliku / interleaved**, pa je supstrat kontrolisana
varijabla. Kad se na tome trenira i radi RL, ispadnu dve ključne činjenice:
vizuelna trajektorija je radna memorija modela (brisanje posrednog *teksta*
jedva da škodi; brisanje posrednih *slika* ruši tačnost 0.638→0.099), i
deterministički scorer tuče frontir VLM sudiju kao RL nagradu — ne samo po
ceni i reproduktivnosti, nego i po *kvalitetu finalne politike* (0.548 naspram
0.508), jer je sudija slepa baš za one modove otkaza koje politika nauči da
eksploatiše. Moja minijatura je reprodukovala drift-skoro besplatno i pokazala
mi gde su tvrdnje rada noseći zid.

## Problem

Benchmarci za vizuelno rezonovanje su *samo za evaluaciju*: možeš da izmeriš
model, ali ne možeš da treniraš na njemu (nema ground trutha na skali) ni da
izvučeš gradijent (metrika nije nagrada). Tri konkretna puta:

1. **Ponuda zadataka.** VBVR-style sintetički setovi daju mnogo *instanci*
   malo *obrazaca rezonovanja* — resempliranje istog generatora je inflacija
   kurikuluma, ne pokrivenost.
2. **Povratna informacija.** VLM-as-a-judge je default nagrada za generisani
   sadržaj, a pokvaren je na tri dokumentovana načina: neprecizna percepcija
   (aditivno mešanje boja — sudije ≥0.80 gde je pravi skor 0.40), previd
   detalja (promenjene veličine prstenova ocenjene 0.78–0.93), pogrešno
   razumevanje zadatka (tačno dopunjen obrazac ocenjen 0.24–0.80). Gore:
   ponovno ocenjivanje *iste sudije na istom izlazu* promeni skor u
   54.6–92.8% slučajeva. Nagrada koja sama sebe re-rolluje je šuman gradijent
   za koji plaćaš API cene.
3. **Nema kontrolisanog poređenja.** Niko nije trenirao video, image i
   interleaved generatore na *istim zadacima sa istim verifierom*, pa je
   „koji supstrat bolje rezonuje" bilo naslućivanje.

## Ideja

Jedna zatvorena petlja, četiri komponente:

```
 300 generatora zadataka ──render──> video | slika | interleaved  (isti problem!)
        │                                   │
   solver + metapodaci               politika (bilo koji generator)
        │                                   │
        └──── deterministički scoreri <─────┘  atributi, ne pikseli
              (soft ponderisana suma × hard gejtovi)
                        │
              RL (Flow-GRPO + CPS sampler)
```

Odluke dizajna koje zaista nose:

- **Generatori, ne datasetovi.** 150 prerađenih VBVR + 150 novih zadataka,
  svaki parametrizovani sampler (veličina mreže, layout, težina) + programski
  solver + metapodaci po instanci (spec, waypointi, ključni atributi). Novi
  zadaci su verifikovano *teži*: medijana 80 distinktnih regiona boja naspram
  12, 47% više-koracnih naspram 7%. Skala: 3.47M slika / 1.30M video; 1.25M
  SFT, 50K RL.
- **Ocena po atributima, ne po razlici piksela.** Lociraj objekte → izvuci
  svojstva (HSV segmentacija, konture, OCR, praćenje trajektorije) → oceni
  svojstva. Robustno na stil, rastavljivo u „koja provera pala". Agregacija:
  soft kriterijumi kao ponderisana sredina; hard ograničenja kao
  *multiplikativni gejtovi* — `score = (Σ wᵢcᵢ / Σ wᵢ) · Πⱼ gⱼ`, tako da 3
  prekršaja zida u lavirintu sa ključem i vratima (zid-gejt 0.73, vrata-gejt
  0.55) ruše skor na 0.37 umesto da se izgube u proseku.
- **CPS (Eq. 1) za eksploraciju.** Group-relative RL treba rollout-e koji se
  razlikuju u *nagradno-relevantnim odlukama*. Eulerov korak
  `x_{t−Δt} = (1−σ)·x̂₀ + σ·[cos(ηπ/2)·x̂₁ + sin(ηπ/2)·ε]`
  premeša predikcioni i svež šum kao **rotaciju** (cos² + sin² = 1), pa
  ukupna magnituda šuma ostaje tačno na rasporedu za svako η. Nasuprot tome,
  Flow-SDE *dodaje* šum preko rasporeda — pri η=0.7 se pozadine doslovno
  posive; nagrada se ruši zajedno sa frejmovima. CPS kupi entropiju odluka po
  ceni nula po fidelitetu.

## Kako radi (intuicija)

**Zašto atributi tuču piksele za ocenjivanje.** Odgovor „zaokruži
neobično-izdvojen" je tačan akko je krug približno prave boje/veličine na
pravom mestu — ne akko frejm poklapa referentni bitmap. Ocenjivanje izvučenih
atributa čini verifier robustan na drift stila i daje *interpretabilan*
ispis po proveri. Podela soft/gejt kodira pravu semantiku: neke provere se
razmenjuju (redosled dolaska naspram tačnosti spajanja), neke su preduslovi
(prolaženje kroz zide invalidira rutu).

**Zašto scorer tuče sudiju za RL.** Dva odvojena efekta, i B.5 papira
hirurškiprecizno pogađa drugi. (a) *Šum*: sudija koja re-scoruje identične
izlaze u 54–93% slučajeva ubacuje varijansu u group-relative advantage.
(b) *Slepilo*: trening RLVLM ruke je dipovao na koracima 400–500 jer je
politika otkrila da suptilno sivenje pozadine ništa ne košta pod VLM sudijom
— ona *gleda* samo objekte. Deterministički scorer proverava boju pozadine
egzaktno i vraća politiku nazad. Slepilo reward-modela nije hipoteza; to je
ekvilibrijum.

**Zašto je vizuelna trajektorija radna memorija.** Kontrafaktičari
(SenseNova-U1, baza 0.640):

| Intervencija | Skor | Čitanje |
|---|---|---|
| parafraza ulaznog teksta | 0.640 | bez promene |
| flip/rotacija ulaza + dosledan tekst | 0.613 | prati *stvarnu* sliku, ne zapamćeni layout |
| ukloni posredni **tekst** | 0.533 | ogreb |
| kontradiktoran posredni tekst | 0.530 | ogreb |
| ukloni ulaznu sliku | 0.064 | kolaps — odgovor nije iz teksta |
| ukloni posredne **slike** | 0.099 | kolaps |
| šum na posredne slike | 0.190 | kolaps |

A u trening ablacijama, zamena celog rezonujućeg *teksta* placeholderima
pomeri skor za −0.004/−0.009, dok kompresija multi-image trajektorije na
jednu sliku košta −0.024/−0.111. Jezik je sigurnosni pojas; frejmovi voze.
Chain-of-Step dekodiranje intermedijatnih denoising latenata rano pokazuje
više kandidatskih ruta kroz lavirint, koje se kasnije konvergiraju —
pretraga hipoteza u samoj denoising trajektoriji.

**RL rezultat (Tab 8, sve ruke uparene osim nagrade).** Baza 0.470 → SFT
0.503 → RLVLM 0.508 → **RLVR 0.548** (OOD: 0.300 → 0.377). RLVR-SDE 0.504.
η sweep: 0.509 / 0.526 / **0.548** (η=0.7) / 0.539; ODE = UniPC 0.522.
RLVR pobeđuje pod *svakim* samplerom. Pre-RL, model commituje pogrešnu boju
u prvim denoising koracima i nikad ne revizuje; posle-RL se ispravlja na
koraku 6 i završava tačno do koraka 29 — isti budžet, bolje *kasnokoračno*
ponašanje.

**Transfer:** Wan2.2-I2V-A14B ruka naspram baze na 7 eksternih suita:
V-ReasonBench 10.2→38.2, MME-CoF-Pro 24.8→48.4, VideoThinkBench-mini
25.7→52.9, RISE-Video 62.8→66.2. NN-analiza pokazuje da su najbliže trening
instance vizuelno/semantički distinktne — transfer ≠ memorizacija.

## Šta sam naučio implementacijom

Petlju sam rebuiltovao kao **MiniVBVR** (`implementation/`): 2 grid-world
generatora (color-sekvencu, lavirint ključ-vrata) sa BFS/L1 solverima,
2.55M-parametarski kondicionalni flow-matching MLP koji „generiše video"
(6-frejm RGB sekvence), scorere sa soft×gejt agregacijom, bg-slep šuman
sudiju i GRPO sa clipped ratio + group-relative advantageima. Radi 38 s na
CPU. Šta me naučio kod:

1. **B.5 drift priča se reprodukuje skoro besplatno.** RLVLM ruka: nagrada
   sudije = puni scorer minus gejt pozadine, plus σ=0.05 šum. Kroz 40
   iteracija fidelitet pozadine monotono driftuje 0.80 → 0.56 a puni skor
   kolabira 0.54 → 0.25 — dok sudija i dalje plaća. Ježev detalj: *soft*
   skor ruke ostaje ~0.70. Politika ne degradira svuda; ona *hirurški*
   eksploatiše slepo mesto sudije. 🎯
2. **RLVR dobit je realna ali, na mojoj skali, tranzijentna** — pik od +0.075
   iznad SFT na iteraciji 10, pa se vraća na SFT nivo. Papirno „replay 60%
   denoising tranzicija" nije fusnota; to je sidro koje sprečava da ranija
   poboljšanja isprve vodom. Moj jednoepohalni GRPO nema takvo sidro. ⚠️
3. **Tvrdnja o sampleru je suptilnija nego što zvuči.** On-schedule svojstvo
   Eq. 1 je egzaktno proverljivo (moj selftest: drugi moment šuma
   1.003/0.996/1.004 za η=0.3/0.7/0.9 — konstantno). Ali motivaciona
   opservacija „ODE grupe su degenerisane → treba CPS" **nije** važila u mom
   toju: moje ODE grupe su imale *više* varijanse nagrade nego CPS (12%
   naspram 38% degenerisanih). Razlog: kod pravog video generatora
   kondicioniranje dominira i kolapsira sve početne šumove na jedan odgovor;
   moj toy MLP propušta početni šum do izlaza, pa je ODE već raznolik.
   Tvrdnja o degeneraciji je svojstvo *klase modela*, ne samplera. ❌→💡
4. **Nereproduktivnost sudije je jeftina.** Dno papirnog opsega flipova
   (54.2% naspram 54.6–92.8%) sam matchovao samo sa σ=0.05 šumom
   kvantizovanim na jednu decimalu. Pola reward signala koji se sam re-rolluje
   je *default* mod otkaza, ne edge case.
5. **Scoreri su tu gde živi inženjering.** Ekstrakcija atributa (dodela
   palete, centroidi po frejmu) je bila više koda od celog GRPO koraka.
   Nagib gejta (0.09/prekršaj, tunirano da 3 prekršaja padne na ≈0.73 kao u
   papiru) je *dizajn odluka* po zadatku — papirno „100 ručno građenih
   scoring programa" je iskreno, i to je pravi cost suite-a.

Tabela presuda (moje naspram papira):

| Tvrdnja | Papir | MiniVBVR | |
|---|---|---|---|
| Deterministički scorer > VLM sudija kao RL nagrada | 0.548 vs 0.508 | 0.542 (pik 0.617) vs 0.246 | ✅ (RLVLM kolabira *jače* na toy skali) |
| Nereproduktivnost sudije | 54.6–92.8% flipova | 54.2% flipova | ✅ |
| Slepilo sudije → drift politike (B.5) | sivenje pozadine, koraci 400–500 | bg 0.80→0.56 kroz 40 iter. | 🎯 |
| CPS > ODE > SDE redosled | 0.548 / 0.522 / 0.504 | 0.546 / 0.541 / 0.534 | ✅ (male margine) |
| RLVR dobit traje do kraja | stabilno +0.045 | tranzijentno, pik +0.075 pa pad | ⚠️ iskreno promašeno (nema replay) |
| ODE grupe degenerisane, CPS daje raznolikost | da | invertovano u toju | ❌ specifično za klasu modela |
| SDE kvari vizuel (bg artefakti) | η=0.7 vidljiva korupcija | nema merljivih artefakata | ❌ toy model se sam počisti na t=0 |

## Šta me iznenadilo / bilo teže nego očekivano

- **Koliko *malo* tekst znači.** Moj prior je bio suprotan — očekivao sam da
  chain-of-thought opisi nose plan. Placeholder tekst koji ništa ne menja
  (−0.004) dok kompresija na jednu sliku košta −0.111 je po meni broj koji
  najviše menja mišljenje u celom radu.
- **Eksploatacija je hirurška, ne opšta degradacija.** Kad sam u sopstvenim
  logovima video soft skor na 0.70 dok pozadina kolabira, B.5 priča je
  postala konkretnija nego iz proze papira.
- **Koliko egzaktno proverljiv Eq. 1 je.** cos²+sin²=1 znači da je
  on-schedule svojstvo selftest od pet linija, ne empirijska nada. Retko se
  vidi RL trik sa dokazivim invarijantom.
- **Papir pošteno otkriva svoje slabe tačke**: težine scorera su ručno
  kalibrisane po zadatku (bez automatizacije), plafon slaganja ljudi je sam
  samo 0.77, vlasnički modeli nisu per-task tunirani, a OOD ovde znači
  held-out *generatore u istom idiomu* — eksterni benchmarci su pravi OOD, i
  tamo su dobiti neujednačene (BabyVision 12.5% apsolutno je još nisko).
  Proverio sam; ablacije i dodatak B.5 su tačno ono što je omogućilo moju
  minijaturu.

## Reference

- Rad: https://arxiv.org/abs/2608.26105 (v2, 2026)
- Moja implementacija: `implementation/` (README sa iskrenim devijacijama,
  results.json sa sirovim brojevima)
- Raščlamba: `breakdown.md` (noseća rekonstrukcija)
- Beleške: `notes.md`
