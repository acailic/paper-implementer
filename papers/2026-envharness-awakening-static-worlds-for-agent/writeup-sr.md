# Writeup (SR) — EnvHarness: Awakening Static Worlds for Agent Learning

> Moje objašnjenje, napisano nakon dva čitanja rada, raščlambe i from-scratch
> mini implementacije frameworka. Nije rezime apstrakta — ovo je šta zaista
> mislim da se dešava, plus šta me naučio kod.
>
> **Rad:** Huang et al., *EnvHarness: Awakening Static Worlds for Agent
> Learning*, arXiv:2608.19880 (2026).

## Verzija u jednom pasusu

Okruženja za treniranje agenata su statična: ponašaju se isto ma koji agent
ušao u njih, pa ne mogu da ciljaju konkretnu slabost konkretnog agenta — a
kad agent jednom reši instancu, nema više čemu da nauči. Očigledno rešenje —
LLM koji generiše nove okoline — zamenjuje pouzdan, ljudima napravljen
verifier hipotetičkim. EnvHarness odbija oba pravca. **Umotava** postojeću
okolinu kroz standardni `reset()/step()/obs()` interfejs — kao što agent
harness omotava zamrznuti LLM — sa tri tipa komponenti (**Stage** = skripta
koja se odigra nad početnim stanjem, **Contract** = filteri nad
akcijama/tranzicijama/opservacijama, **Chain** = dve okoline spojene
serijski), i nikad ne dira kod dinamike ni nagradu. Nagrada se **nasleđuje
konstrukcijom**, pa se metrika ne može igrati njome. Automatska petlja
(**EnvRigger**) posmatra rollout-e ciljne politike, dijagnostikuje greške,
piše kandidate i prihvata ih samo ako sveži rollout-i padnu u „rešivo ali
izazovno" opseg uspešnosti. Rezultat: ciljani kurikulumi nad istim poverljivim
benchmarkom — do +9 poena u skill-učenju i pobede u RL-u, preko četiri domena.

## Problem

Svaka ranija linija guta jedno od dva zrna:

| Pristup | Šta radi | Zašto puca |
|---|---|---|
| GenEnv-style ko-evolucija | LLM simulira tranzicije + provere uspeha | Halucinirana dinamika; verifier više nije onaj poverljivi |
| EnvGen-style editovanje konfiguracije | Krpi mape/logiku unutar simulatora | Duboka operacija po benchmarku; može da slomi logiku stanja |
| Agent-World-style sinteza | Generiše toolsetove/baze/zadake od nule | Izmišljeni kriterijumi ocenjivanja; ogroman inženjering |
| PAIRED / level replay (klasični RL) | Adversary ili prioriteti repleja biraju nivoe | Traži parametrizovan prostor nivoa; ne prenosi se na otvorene zadatke |

Ili **zameniš** poverljivu okolinu (i izgubiš validnost) ili je **forkuješ**
po benchmarku (i izgubiš opštost). Tvrdnja rada: postoji treća opcija koja
stoji pred nosom — sam gym interfejs je dovoljno izražajan da preoblikuje
okolinu *spolja*.

## Ideja

Tri stvari, i srednja je pravi doprinos:

1. **Okolina je 6-torka** E = (S, A, O, T, R, s₀). *Komponenta* je
   dekorator na nivou interfejsa w(E) = E′ koji sme da prepiše bilo koji
   podskup od (S, A, O, T, s₀) — **nikad R**. To nije propust nego dizajnerska
   odluka: „uspeh je verdict samog benchmarka, pa preoblikovanje nagrade ne
   može da pomeri metriku."

2. **Tri tipa komponenti pokrivaju celu torku jednim mehanizmom:**
   - **Stage** — novo početno stanje = kratka skripta akcija δ odigrana nad
     čistim s₀ posle reset-a. Elegantno: mutirani start je *po konstrukciji
     dostižno stanje* originalne dinamike. Nema ilegalnih stanja, nema
     ručnih šema.
   - **Contract** — tri čiste funkcije: `filter_action` (prepiši ili
     blokiraj), `modify_transition` (prepiši odgovor koraka),
     `filter_observation` (transformiši šta agent vidi). Blokirana akcija
     ne dira okolinu i vraća *razlog* blokade — odbijanje nikad ne ostavlja
     politiku u ćorsokaku.
   - **Chain** — dve cele okoline spojene u jedan epizodni tok; kompozitni
     verifier je konjunkcija R_A ∧ R_B.

   Slaguju se: E′ = w_chain ∘ w_contract ∘ w_stage (E), i **ne
   komutiraju** — Stage unutar Contract-a znači da se i sama replay skripta
   filtrira. Red gnežđenja je semantika.

3. **EnvRigger automatizuje dizajn.** Observe (rollout-i ciljne politike,
   uspeši *i* neuspesi) → Diagnose (petlje ponavljanja, parsing greške,
   pogrešno pročitana ograničenja alata; smer zavisi od toga da li politika
   zapinje ili odlično rešava) → Write (kandidati, često Stage + Contract
   zajedno) → Validate (sveži rollout-i; prihvati ako SR padne u opseg).
   Validacija nosi tri pravila koja zaslužuju više zahvalnosti:
   - **Baseline prvo**: K nemutiranih rollout-a kažu da li pooštravanje uopšte
     ima smisla (baseline SR ≈ 0 → podupri, ne otežavaj), koliko ima
     prostora i koje delove okoline politika stvarno koristi.
   - **Pitfall**: SR = 0 iz *nemogućnosti* je isto beskoristan kao SR = 1 iz
     trivijalnosti. Ako većina rollout-a timeout-uje, sledeći predlog mora
     da se povrati ili olabavi — „slaganje još zabrana ne može da se penje u
     opseg."
   - **Refine**: ako se SR pomerio *ka* opsegu, zadrži hook-ove verbatim i
     diraj samo magnitudu; ako se nije pomerio, promeni tip perturbacije.

Dizajner-LLM deli backbone sa politikom — nema jačeg učitelja, pa dobici
nisu prikrivena destilacija.

## Kako radi (intucija)

Ono što mi je kliknulo: **opseg prihvatanja radi posao kurikuluma, a
rollout-i radi posao kritičara.** Niko ne ocenjuje generisanu komponentu
direktno — niko joj čak ni ne gleda kod. Jedino pitanje je: „da li trenutna
politika, ubačena u omotanu okolinu, ponekad uspeva i ponekad ne uspeva?"
Komponenta koja prođe taj test je, po definiciji, zadatak natenčano na ivici
sposobnosti ovog agenta. Zato se petlja ponavlja po batchu od 50 okolina:
opseg je definisan relativno na *trenutnu* politiku, pa se kurikulum
ko-evoluira.

Druga intuicija: **dostižnost-po-replay-u je ono što čini Stage bezbednim.**
Klasični propust „samo postavi početno stanje" su ilegalna ili degenerisana
stanja. Odigravanje skripte akcija može da proizvede samo stanja koja
stvarna dinamika proizvodi. Cena: reset-i moraju biti deterministički — što
je i iskrena granica rada (bez živih servisa, pravih korisničkih naloga,
fizičkih robota).

Treće: **zaključana R-osa je priča o poverenju.** Svaka metoda generisanja
okolina na kraju mora da odgovori na „zašto da verujem tvom verifier-u?"
Odgovor EnvHarness-a je strukturalan: omotač fizički ne može da promeni
nagradu, jer nagrada nije među njegovim parametrima.

## Šta sam naučio implementacijom

(Iz `implementation/` — `ActionableEnv` + KeyGrid bridge + Stage/Rules/Link
dekoratori + statistički EnvRigger + tabular Q-learning politika, čisti
stdlib, ~35 s. Detalji u `implementation/README.md`.)

1. **Statistika sondiranja JE petlja.** Sa K = 3 dijagnostička rollout-a po
   zadatku, zadatak sa stvarnim SR = 0.5 se ~25% vremena pogrešno
   dijagnostikuje kao savladan/beznadežan; prihvaćeni skup se nasumično
   meša i „ciljanje" se raspline u šum — moj EnvHarness krak je postao
   statistički neodvojiv od nasumičnog omotavanja. Petlja se stabilizovala
   tek na K = 8–10. Rečenica „odlučuj iz statistike rollout-a, nikad iz
   jednog trace-a" je nosiva na način koji ablacija nikad ne kvantifikuje.

2. **Enkodiranje stanja dominira nad kurikulumom.** `(pozicija, koje-ključevi)`
   je Markovljevsko ali specifično za zadatak → nula transfera.
   `(pozicija, n_ključeva)` je deljeno ali ne-Markovljevsko → trening kolabirao
   na ~0. Transfer je omogućilo tek *lokalno reaktivno* enkodiranje (maska
   zidova, broj ključeva, smer ključa, smer cilja). Rad ovaj problem nema jer
   LLM čita opservacije u kontekstu; lekcija: ciljani kurikulumi se isplate
   samo ako klasa funkcija learns-era može da ih konzumira.

3. **Contract-i se tuku sa bezmemorijskim learner-ima.** Contract koji zabranjuje
   skok je nevidljiv tabularnom stanju, pa preplitanje blokiranih i
   neblokiranih okolina daje kontradiktorne Q-update-e (udeo skoka je titrao
   0.0 ↔ 1.0 kroz seed-ove). Stage komponente — čisti pomeraji početnog
   stanja — taj problem uopšte nemaju. U radu oba tipa pomažu LLM politici;
   u mojoj igrački samo Stage preživljava susret sa Q-tabelom.

4. **Nekomutativnost je stvarna i jeftina za demonstraciju.** `Rules(Stage(env))`
   odigra skriptu sirovo; `Stage(Rules(env))` filtrira samu skriptu (2/2
   akcije blokirane, drugačije startno stanje). Test od pet linija dokaže
   jed. 5 ubedljivije od pasusa.

5. **Konjunkcija verifier-a i crash-safety hook-ovi se pokazuju u praksi.**
   Link-ova R_A ∧ R_B semantika (sabotiraj pod-zadatak B, gledaj kompozit kako
   pada na fail dok A i dalje prolazi) i `step_reward` exception gard (dense
   hook baca izuzetak svaki korak → epizoda se ipak završava, 22 greške
   zabeležene, nijedna fatalna) — oba su pala iz raščlambe tačno kako su
   napisana.

## Šta me iznenadilo / bilo teže nego što sam očekivao

**Glavni poredak rada se nije replikovao na mojoj skali — a iskren razlog je
zanimljiv.** Četiri seed-a, 15k epizoda po kraku, evaluacija na held-out
*originalnim* zadacima:

| krak | SR open | SR hard | SR sve (min [min–max]) | prosečni koraci |
|---|---|---|---|---|
| original (statični pool) | 0.75 | 0.02 | 0.39 [0.24–0.52] | 16.4 |
| **envharness (band-validiran)** | 0.77 | 0.00 | 0.39 [0.34–0.47] | 14.9 |
| randomwrap (bez validacije) | 0.85 | 0.01 | 0.43 [0.38–0.51] | 14.9 |

- Srednja vrednost se ne pomera, ali **varijansa se smanjuje**: najgori seed
  0.34 naspram 0.24. Opseg garantuje da nema degenerisanog trening batcha, pa
  se podiže pod čak i kad plafon stoji.
- Poduprti džepni zadaci *se* pale u treningu (7–12 od 16 prihvaćeno po
  batchu), ali transfer na hard evaluaciji ostaje blizu nule — scaffold
  trenira sufiks trajektorije; nepoduprti prefiks se nikad ne poseti.
- Moj `randomwrap` baseline ispostavio se **prejak da bi bio fer dvojnik**
  „generisanih okolina" iz rada: svaki nasumični Stage spec u mom meniju
  drži zadatak rešivim za verifier, pa mu fali baš onaj failure mode
  (pokvarena okolina) protiv kog write-and-validate postoji. Napraviti
  *loš* generator je lako; napraviti ga lošim *na pravi način* — nije.

Takođe iznenađenje: koliki deo inženjeringa rada živi u
**`get_env_state()` runtime-safe pogledu** — samo obični podaci, bez Docker
handle-a i browser stranica — što je ono što čini LLM-pisani kod komponenti
prenosivim između puzzle u memoriji i kontejnerizovanog repoa. Deluje kao
fusnota; zapravo je cela priča o portabilnosti.

## Verdikt

Sloj frameworka (ActionableEnv + Stage/Contract/Chain + mehanika petlje) se
čisto reimplementira i svaki mehanizam se ponaša kako je opisano na nivou
jedinice. *Perfomans* tvrdnja traži learner-a čija klasa funkcija ume da
konzumira ciljane kurikulume — LLM koji čita opservacije. Tabularni learner
zasiti dobit na „manje bačenih epizoda, manja varijansa" — što je i dalje
nešto, samo nije +9 poena.

## Reference

- Rad: https://arxiv.org/abs/2608.19880
- Zvanični kod: https://github.com/google-research/envharness
- Moja implementacija: `implementation/` (vidi `implementation/README.md`)
- Raščlamba (rekonstruiše metodu od nule): `breakdown.md`
- Beleške sa čitanja: `notes.md`
