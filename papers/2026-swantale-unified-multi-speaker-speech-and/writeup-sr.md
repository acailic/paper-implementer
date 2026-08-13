# Pregled — SwanTale: Objedinjena generacija višegovornog govora i zvuka

> **Rad:** SwanTale: Unified Multi-Speaker Speech and Audio Generation for Instruct and Zero-Shot Tasks
> **Autori:** Yu Zhang, Ruiqi Li, Changhao Pan, Ke Lei, Xiang Yin, Cheng Yang (ByteDance / Zhejiang University)
> **Godina:** 2026 · **ArXiv:** https://arxiv.org/abs/2608.02023

Ovo je moje lično objašnjenje, napisano nakon čitanja i ponovne implementacije
rada. Ovo je sinteza, ne prepričavanje apstrakta.

## Verzija u jednom pasusu

SwanTale je industrijski razmeran generator zvuka iz ByteDance-a koji radi dve
stvari koje inače rade *različiti* modeli — **osmišljavanje** glasa i scene iz
tekstualnog opisa (*instruct* zadatak) i **kloniranje** glasa iz referentnog
snimka (*zero-shot* zadatak) — unutar jednog ne-uzročnog flow-matching
Transformer-a. Elegantni potez je u tome što se ta dva zadatka razlikuju samo u
dve stvari: *šta* se od opisa ubaci i *koji frejmovi latenta* se maskiraju kao
fiksni referentni kontekst. Ostatak rada je kula inženjeringa koji ovo čini
izvodljivim na 2 milijarde parametara: glatki 25 Hz audio VAE (SwanVAE),
heširano n-gram memorija za opise (Engram sloj), dual-router Mixture-of-Experts
koji troši računanje samo gde talasnog oblik to zahteva, trik koji pretvara
„neka bude visokog kvaliteta" u uslovnu promenljivu umesto u problem
potvrđivanja kroz RL, i flow-matching GRPO faza čiji prosečni log-verovatnoća po
elementu drži odnos važnosti blizu 1 bez obzira na dužinu izgovora.

## Problem

Kreator medijskog sadržaja — sinhronizacija animacije, reklama, audio drama,
montaža kratkog videa — želi *jedan* sistem koji može da:

1. **Izumne glas iz opisa** (bez referentnog snimka): *„stariji muškarac govori
   toplo u tihoj kafani sa dalekim zveckanjem šoljica."* To je **instruct**
   zadatak. Opis navodi ličnost govornika, akustičko okruženje i finu sadržinu.
2. **Klonira i ponovo iskoristi glas** iz kratkog snimka — **zero-shot**
   zadatak, klasičan TTS — uz podršku za dijalog više govornika.
3. Smesti govor *unutar* akustičke scene sa pravovremenim negovornim zvučnim
   efektima (koraci, vetar, muzički ubodi), sve u jednom talasnom obliku umesto
   da se spaja naknadno.

Postojeći sistemi pokrivaju po jedan od ovih. Instruct-TTS daje samo govor (ne
scene) i ne ume da klonira; zero-shot-TTS ne prihvata opise i ne ume da osmisli
glas iz osnove; a nadogradnja govora + zvučnih efekata u nizvodnom procesu
proizvodi drift u vremenu, reverberaciji i jačini. Autori imenuju i tri prave
prepreke: **oskudica podataka** (bogati višeslojni opisi su skupi za anotiranje),
**saglasnost zadataka** (uslovljavanje opis→stil i audio→stil se međusobno
slabe pri zajedničkom treniranju) i **složenost više audio modalnosti** (govor,
efekti, pevanje i muzika imaju vrlo različite vremenske strukture i moraju
koegzistirati).

## Ideja

Jedna najvažnija ideja je ujedno i najjednostavnija: **ujedinjenje zadataka
putem maskiranja.** Uzmi čistu latentnu metu `x★`. Uzorkuj flow-matching vreme
`t` i šum `ε`, formiraj šumoviti latent `x_t = (1−t)ε + t·x★` i neka model
predviđa brzinu `v = x★ − ε`. Zatim:

- Za **instruct** zadatak, maska `m = 0` svuda (generiši sve frejmove) i ubaci
  *puni* opis (okruženje + govornici + sadržina).
- Za **zero-shot** zadatak, maskiraj *referentne* frejmove `m = m_prompt`
  (zadrži ih na čistoj meti, tako da deluju kao uslovniprefiks) i ubaci samo
  opis *sadržine*.

Jedna mreža (backbone), jedan cilj brzine, dva zadatka koji se razlikuju samo u
sadržaju opisa i tome koji su frejmovi maskirani. To je čitav trik ujedinjenja i
zaista je čist.

Ostalo je inženjering koji ovo čini praktičnim i cedi kvalitet:

- **SwanData-Caption** — proces podataka (~70M opisnih zapisa) koji čisti audio
  iz medija, radi vokalnu separaciju / diarizaciju / ASR / poravnanje, i koristi
  multimodalni LLM da anotira tri strukturirana polja (Okruženje / Govornici /
  Sadržina), rafinisano filtrima kvaliteta i pametnim grupnim poređenjem
  najbolji–najgori po izražajnosti umesto apsolutnog MOS bodovanja.
- **SwanVAE** — asimetrični audio VAE: čisto-CNN enkoder snažno poduzorkovan
  (48 kHz → 25 Hz), Gausov usko grlo i SAME-style Transformer resampling
  dekoder. Daje glatke kontinualne latentte koje je lako modelovati flow
  matching-om.
- **Engram sloj** — heširana n-gram memorija (reda 2 i 3) prikačena na
  ugradnje opisa, sa sadržajno-zavisnim sigmoidnim kolom čiji je bias
  inicijalizovan *negativno* tako da memorijska putanja polazi zatvorena i
  otvara se samo za strukturirane markere.
- **Unified MoE** — dual router koji zamenjuje svaki drugi DiT FFN: *task-level*
  router bira deljene eksperte, a *frame-level* audio router koristi dinamički
  Top-P izbor sa null ekspertima i vremenski-svesnim budžetom `q(t)` tako da se
  računanje troši gde talasnog oblik to zaista zahteva. Balansiranje opterećenja
  je *bez pomoćnog gubitka* (podešavanje bias-a), stabilizovano z-gubitkom i
  kaznom za null-kolaps.
- **Uslovljeni kvalitet kroz nagradu** — umesto RL, kvalitet (STOI / SI-SDR /
  PESQ / MOS) se ubacuje kao opis + zastavica, a pri zaključivanju se zastavica
  **forsira na „visoko".** Težak problem postaje problem uslovne generacije bez
  ijednog rollout-a.
- **Flow-matching GRPO** — kada se RL *ipak* koristi (izgovor, slaganje atributa,
  sličnost govornika), definišu marginalno-očuvajući SDE tako da je svaki Euler
  korak Gausov, i koriste **prosečnu log-verovatnoću po elementu** tako da
  odnos važnosti ostane blizu 1 bez obzira na dužinu izgovora, sa
  zatvoreno-formom KL prema smrznutoj referentnoj politici (ista varijansa →
  samo član sa kvadratom sredine).

## Kako radi (intuicija)

Zamisli da talasnog oblik zvuka SwanVAE kompresuje u glatki 25-frejm-po-sekundi
latent. Flow-matching generator zatim uči *polje brzine* koje gura čisti šum
(t=0) ka čistim podacima (t=1). Lukavost je u tome kako dva zadatka dele to polje:

- Instruct = „denoizuj sve, vođeno punim opisom."
- Zero-shot = „denoizuj sve *osim* čistog referentnog prefiksa, koji sidri
  identitet, vođeno samo sadržinom."

Pošto su referentni frejmovi držani čistim i spojeni u šumovito stanje, model
ih vidi kao istinit kontekst — ne mora da *rekonstruiše* identitet, samo da ga
*nastavi*. Zato jedan cilj dovoljan.

Unified MoE zatim odlučuje, po frejmu, koliko *dodatnog* računanja taj frejm
zaslužuje. Frejm tišine rutira ka null ekspertu (jeftino); bogati voicovani
frejm rutira ka više audio eksperta (skupo). Vremenski-svesni budžet `q(t)`
pomerа ovo kroz flow vreme `t` — više kapaciteta rano u denoizingu (struktura)
nego kasno (tekstura). A Engram sloj daje grani opisa jeftinu tablicu za
pretragu tako da strukturirane fraze („tiha kafana") dohvataju naučene ugradnje
umesto da se izvode iznova svaki put.

Pri zaključivanju pokreću dvostepeno dekomponovano classifier-free guidance —
null → tekst+govornik → puno — sa sway mrežom uzorkovanja `t(u) = 1 − cos(πu/2)`
koja stavlja više koraka integracije pri šumovitom kraju.

## Šta sam naučio implementirajući

Implementacija četiri samostalne celine (Engram, Unified MoE, flow DiT sa
maskiranjem zadataka i sway uzorkovanje) na sintetičkim latentima učinila je
nekoliko stvari konkretnim koje rad pominje samo usput:

1. **Ujedinjenje maskiranjem zaista „prosto radi" kao jedan cilj.** U mojoj toy
   probi, i `inst` i `zero` zadaci trenirani su pod jednom deljenom mrežom, a MSE
   na frejmovima generacije pao je dobro ispod početnog nivoa šuma za oba. Ne
   treba poseban zero-shot gubitak — split maske + opisa je čitav mehanizam.
2. **Negativno-inicijalizovan bias kola na Engram sloju je važan.** Sa nultim
   ili pozitivnim bias-om, memorijska putanja odmah se aktivira i može rano
   dominirati / destabilizovati trening; negativan bias čini od nje *rezidual
   koji mora zaslužiti svoju aktivaciju*, što se slagalo sa tvrdnjom rada, ali
   je bilo opipljivo tek kad sam posmatrao gubitak.
3. **Balansiranje opterećenja bez pomoćnog gubitka je iznenađujuće efikasno na
   igrački.** Puko pomeranje bias-a izbora svakog rutiranog eksperta za
   `η·(1/R − f_i)` — bez pomoćnog gubitka verovatnoće dodele — držalo je moju
   upotrebu eksperta blizu uniformne, a kazna za null-kolaps (`L_null`) zaista
   je sprečila da se svi tokeni slome na null ekspertu. z-gubitak je držao
   logite rutera da ne lutaju.
4. **Trik prosečne log-verovatnoće po elementu je prenosivo blago iz GRPO
   steka.** Pošto svaki latentni element doprinosi jednu log-verovatnoću i
   ih *prosečavaš*, odnos važnosti `ρ = exp(ℓ_θ − ℓ_old)` ostaje O(1) nezavisno od
   dužine sekvence — varijanta sa sumom eksplodira za duge izgovore. A pošto
   SDE tranzicija deli istu varijansu za politiku i referencu, KL se urušava na
   čist član sa kvadratom sredine. Ove dve ideje generalizuju se daleko van zvuka.
5. **Nagrada-kao-uslov je tiho moćna ideja.** Forsiranje zastavice kvaliteta na
   „visoko" pri zaključivanju je trik od jedne linije koji zaobilazi čitavu RL
   petlju. Nisam implementirao pun GRPO, ali *uslovno* uokviravanje je nešto čega
   bih se latio u bilo kom uslovnom generatoru.

## Šta me iznenadilo / bilo teže nego očekivano

- **Koliko malo rad izoluje svaku komponentu.** Engram, dual router, uslovljeni
  kvalitet, kurikulum, GRPO su svi naslagani zajedno, i ablacije ih ne razdvajaju
  potpuno — tako kad nešto radi, ne možeš uvek reci *koja* dugmad je zaslužna.
  To je realnost industrijskih radova, ali je učinilo „šta da reimplementiram?"
  pitanjem prosuđivanja. Izabrao sam četiri celine koje su algoritamski
  samostalne.
- **Dinamički Top-P + vremenski-svesni budžet je trapav.** Dobiti budžet `q(t)`,
  Top-P prag, null bias i faktor kapaciteta da se *zajedno* pomeraju kao funkcije
  istog skalara — i trenirati Gumbel temperaturu nadole bez lomljenja rutiranja
  — zahtevalo je najviše pažnje. Podela „izbor koristi bias-prilagođene logite,
  težine koriste bazne logite" lako se obrne.
- **Sway uzorkovanje zaista menja rezultate.** Neuniformna mreža zvuči kao
  kozmetički detalj, ali koncentrisanje koraka pri `t=0` (gde je polje
  najstrmije) merljivo je popravilo kvalitet toy uzorka naspram uniformnih
  koraka pri istom broju koraka.
- **Razmera je prava priča.** 2B aktivnih parametara, 64×A100, ~70M opisnih
  zapisa i Qwen enkoder za opise znače da je *proces podataka* (SwanData-Caption)
  verovatno najveći doprinos i najmanje reimplementabilan. Glavna ablacija —
  skaliranje enkodera opisa 8B→32B najviše diže tačnost instrukcije
  (3.39→3.70) — potvrđuje da je kapacitet razumevanja opisa pravo usko grlo, koliko
  i sam akustički generator.

## Reference
- Rad: https://arxiv.org/abs/2608.02023
- Moja implementacija: `implementation/` (`model.py`, `train.py`, `data.py`, `README.md`)
- Rastav (puna metoda, matematika, arhitektura): `breakdown.md`
- Duboke beleške (dvoprolažno čitanje): `notes.md`
