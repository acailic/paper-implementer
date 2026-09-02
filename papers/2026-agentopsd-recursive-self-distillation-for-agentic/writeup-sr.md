# Writeup (srpski) — AgentOPSD: rekurzivna self-distillacija za agentski RL

> Rad: "AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement
> Learning" — Wang et al., 2026, arXiv:2608.05987.
> Ovo je moja sinteza posle čitanja i implementacije, ne prepričavanje
> apstrakta.

## Verzija u jednom pasusu

Kad trenirate LLM agenta sa GRPO na multi-turn task-u, okruženje vam na kraju
epizode da jedan bit reward-a, a GRPO taj jedan skalar razmaže preko svakog
tokena trajektorije. Potez AgentOPSD-a je da prestane da tretira taj skalar
kao jedini signal: pokrene *iste težine politike* drugi put sa dohvaćenim
"skill"-om pridodatim kontekstu (privilegovani teacher), uzme per-token
log-prob razliku između ta dva prolaza, sabere je po turnu u "evidenciju",
tu evidenciju rekurzivno uvuče u bayesovsko uverenje (u log-odds
prostoru) da će trajektorija uspeti, i dodeli svakom turnu *graničnu
reviziju uverenja* koju je izazvao — sa predznakom stvarnog ishoda. Ta
per-turn kreditna signal onda moduliše — ali nikad ne može preokrenuti —
GRPO advantage. Bez kritika, bez dodatnih rollout-a, jedan dodatni forward
pass.

## Problem

Dodela kredita (credit assignment). U 30-turn ALFWorld epizodi možda 2–3
turna odluče ishod (pravi objekat uzet, prava pretraga izdata); ostalo je
rutina. GRPO to ne vidi: advantage se računa jednom po trajektoriji iz
statistike uspešnosti grupe, pa se emituje svakom tokenu. Ključni signal se
razblažuje sa 27 turnova šuma, i rad meri posledicu direktno — GRPO gubi
~2,9 poena uspešnosti po dodatnom turnu, AgentOPSD ~0,5. Klasične popravke
sve nešto koštaju u settingu koji to teško plaća: PPO traži naučenog
kritika, VinePPO rollout-e iz intermedijernih stanja, process reward modeli
superviziju. Postojeća jeftina alternativa — privilegovana
self-distillacija (pokreni politiku sa hint-om, uporedi log-prob-ove) — daje
signal koji je *token-lokalan i turn-kratkovid*: ocenjuje svaki token
izolovano i ignoriše i to da okruženje odgovara samo na cele akcije, i
evidenciju koju su raniji turnovi već akumulirali.

## Ideja

Lokalna teacher–student log-prob razlika nije kredit. **Kredit je koliko
evidencija jednog turna pomeri tekuće uverenje da će trajektorija uspeti.**
Zato: agregirati razlike po turnu (cele akcije su ono što okruženje vidi),
držati uverenje kao log-odds akumulator (gde se nezavisna evidencija samo
sabira), i platiti svakom turnu *promenu* uverenja koju je izazvao, a ne
njegovu sirovu evidenciju. Dve elegantne posledice padaju iz matematike
besplatno:

1. **Nesigurnosni gate.** Blizu sredine sigmoide evidencija mnogo pomeri
   uverenje; kad se uverenje zasiti blizu 0 ili 1, ista evidencija pomeri
   ~ništa (izvod B(1−B) iščezava na krajevima). Redundantna potvrda u već
   odlučenoj epizodi automatski zaradi ~nula kredita — tačno ono što "sirov
   gap" kredit nema.
2. **Sigurnost.** Preoblikovani advantage živi u ograničenom opsegu oko
   GRPO advantage-a — `(1−λ)A` do `(1+λb)A` — tako da metod može modulisati
   magnitudu, ali nikad preokrenuti smer update-a. Drop-in na svaki GRPO/PPO
   stack, i λ=0 vraća tačno čist GRPO.

## Kako radi (intuicija)

Zamislite bookmakera koji ažurira kvote posle svakog turna epizode. Početna
linija je stopa uspešnosti grupe — "tekuća politika rešava ovaj task 30%
vremena" — što je daleko bolji prior od 0,5. Svaki turn bookmaker pročita
jedan komad evidencije: *da li se akcija koju skill podržava poklopila sa
onim što je politika stvarno uradila?* (izsumirana log-prob razlika).
Evidencija se akumulira aditivno u log-odds sa geometrijskim opadanjem γ, i
kredit svakog turna je promena bookmakerove citirane verovatnoće. Na kraju
sudija (verifier) otkrije ishod: na uspehu, revizije naviše su bile
proročanske; na neuspehu, iste te revizije naviše su bile varljive — otud
`q_k = sign(A)·ΔB_k`. Standardizovati unutar trajektorije, kliknuti u
multiplikativni opseg, pomešati sa uniformnim GRPO advantage-om, pokrenuti
standardni clipped update.

Jedan iskren hand-wave (o kojem rad otvoreno govori): ceo lanac pretpostavlja
*ponašanje uslovljeno skill-om ≈ ponašanje uslovljeno uspehom*. Ako je
dohvaćeni skill loš, "evidencija" meri skill endorsement, ne verovatnoću
uspeha. Rad dokazuje da pristrasnost proksija ne može preokrenuti znakove i
rangove *pod* tom pretpostavkom — ali nikad ne izvodi skill-quality
ablaciju koja bi testirala samu pretpostavku. To je pravi otvoreni propust.

## Šta sam naučio implementacijom

(Iz `implementation/` — GRU-skala politika na sintetskom pivotal-turn task-u
gde je ground-truth per-turn kredit *poznat*, što realni benchmark-i ne mogu
da ponude.)

- **"Self-distillacija" je samo dva forward prolaza iste mreže.** Pisanje
  koda je razbilo mistiku: teacher su tekuće težine, `sg`-detached, sa
  nekoliko dodatnih kontekstnih tokena. Cela privilegovana mašinerija je
  jedan dodatni forward i elementwise oduzimanje.
- **Prior uverenja je nosiv, a skoro besplatan.** `B₀` = stopa uspešnosti
  grupe koju ste već izračunali za GRPO. Sidrenje log-odds tamo (umesto na
  0,5) vredi −10,2 poena u ablaciji rada — najveća pojedinačna komponenta.
  Košta nula dodatnog računanja.
- **Graceful-degradation putanja je stvarna.** Sa K=1 turnova
  standardizacija daje z=0, dakle w=1, dakle tačno GRPO. Moja γ ablacija
  (0,5/0,8/1,0) je takođe pokazala da konstanta opadanja jedva igra ulogu na
  kratkim horizontima — u skladu sa nalazom samog rada da osetljivost na
  hiperparametre kolapsira na kratkohorizontnim taskovima.
- **Evidencija turna raste sa dužinom turna.** `e_k = Σ_t δ_t` raste sa
  brojem tokena po konstrukciji, pa duže akcije akumuliraju više
  "evidencije" po turnu — konfound koji rad ne normalizuje. Na to sam
  naleteo kad sam odlučivao kako da brojim skill blok.

## Šta me iznenadilo / bilo teže nego očekivano

- **γ=1 je bukvalno Waldov SPRT (1945).** Rekurzivno uverenje bez opadanja
  je sequential probability ratio test, star osam decenija, u novom RL
  ruhu. Doprinos rada je *konverzija* lokalnog signala u sekvencijalni
  kredit — ne akumulator sam.
- **Redosled ablacija.** Očekivao sam da će rekurzija (po kojoj rad nosi
  ime!) biti najveći doprinos, ali uklanjanje predznaka ishoda (−8,6) i
  empirijskog priora (−10,2) boli daleko više od raw-gap-umesto-ΔB (−6,3) i
  per-token granularnosti (−3,2). Smer i sidrenje na stanje važe više od
  mehanizma rekurzije.
- **Na mom toy task-u, čist GRPO je konvergirao *brže*.** Obe ruke su
  dostigle 1,000 uspešnosti, ali GRPO 0,8 do iteracije 10 naspram 0,3 za
  AgentOPSD na težem probu. Razlog je poučan: moj privilegovani skill je
  mala lookup tabela koju GRU internalizuje za nekoliko iteracija, pa je
  gust signal ishoda dovoljan i preoblikovani advantage-i uglavnom dodaju
  rani šum. Korist koju AgentOPSD tvrdi živi u dugohorizontnim,
  sparse-success režimima koje CPU toy ne može da izrazi. Ono što run
  *potvrđuje*: svaka jednačina pipeline-a se izvršava, trenira stabilno i
  dostiže optimalno ponašanje.
- **Lokalizacija kredita je teška za merenje čak i sa oraklom.** Moj
  Spearman-vs-oracle metrik je bio šuman preko seed-ova u toy skali —
  quantity koje rad *stvarno* optimizuje uočljivo je samo agregatno.

## Reference

- Rad: https://arxiv.org/abs/2608.05987
- Zvanični kod: https://github.com/ZethWang/AgentOPSD
- Moja implementacija: `implementation/` (model.py, data.py, train.py —
  `python3 train.py`, ~2 min CPU; iskreni rezultati u `run_output.txt`)
- Breakdown: `breakdown.md`
