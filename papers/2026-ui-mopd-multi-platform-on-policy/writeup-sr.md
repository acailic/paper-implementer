# Pisanje — UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning

> Papir: Lian et al., "UI-MOPD: Multi-Platform On-Policy Distillation for
> Continual GUI Agent Learning", arXiv 2607.04425 (2026).
> Moje objašnjenje, kao da ga predajem kolegi koji nije čitao papir.

## Verzija u jednom pasusu

UI-MOPD trenira jednog zajedničkog GUI agenta da radi i na desktopu i na
mobilnom, bez da ponašanje jedne platforme obriše drugu. Radi se u dve faze:
prvo se SFT-om istreniraju dva velika odvojena *učitelja*, svaki specijalizovan
za jednu platformu na novom skupu podataka (Uni-GUI); zatim se trenira jedan
manji *učenik* on-policy RL-om, gde se u svakom koraku učenikov rollout
usmerava odgovarajućem učitelju platforme i blago vuče ka njemu preko
ne-negativnog K3 KL estimatatora, dok rule-based ishodna nagrada vodi ka uspehu
zadatka. KL kazna je kontrolisana *adaptivnom maskom* koja odbacuje učiteljevo
ograničenje čim grupa promptova već dovoljno dobro zarađuje nagradu, pa je
učenik tamo slobodan da istražuje, a ostaje usidren tamo gde mu ide teško.
Rezultat: 8B učenik pobeđuje svoj 32B osnovni model na MobileWorld *i* napreduje
na OSWorld — nešto što naivni mixed-SFT i model merging ne uspevaju.

## Problem

GUI agenti su prešli sa "radi na jednoj platformi" na potrebu da rade na više
njih — desktop, mobilni, web — jer korisnici ne žele posebnog agenta po
uređaju. Ali kontinualno učenje kroz platforme nailazi na dva zida:

1. **Retkost podataka.** Dobri cross-platform trajektorije su retke. Većina
   postojećih skupova je za jednu platformu, a ono što postoji je bučno:
   nevažeće akcije, pogrešno poravnate parove stanje-akcija, nedosledna
   granularnost zadataka.

2. **Mešanje konvencija ponašanja.** Desktop i mobilni imaju *različitu*
   semantiku akcija. Zatvaranje prozora vs. dugme back; prevlačenje mišem vs.
   dvoprstni swipe; `key` događaj vs. `system_button`. Naivno slivanje podataka
   i treniranje jednog modela — kroz mixed SFT, mixed RL, ili weight averaging
   / TIES merging posebno istreniranih checkpointa — daje *prosečnu* politiku.
   Gore, u kontinualnom učenju novo-naučena platforma katastrofalno zaboravi
   staru. Tabela 2 iz papira je direktna: 8B model istreniran SFT-om samo na
   desktopu poboljšava OSWorld (33.9 → 35.8) ali obara MobileWorld na **0%**.

Zato je pitanje: kako zadržati jedan skup težina koji je *dobar na obe*
platforme, a ne samo prosek koji je osrednji na obe?

## Ideja

Zadrži stručnost specifičnu za platformu, ali je preseli. Umesto da pokušavaš
da uguraš obe platforme u jedan model preko mešanja podataka (što konfliktuje)
ili spajanja težina (što je destruktivno), **istreniraj dva odvojena velika
učitelja, svaki ekspert za jednu platformu, pa ih destiluj u jednog manjeg
učenika tokom onlajn RL-a — usmeravajući svaki rollout njegovom učitelju
platforme po uputstvu.**

Učitelji su smrznuti bihejvioralni sidrovi. Učenik je jedna politika koja
uči, u deljenom prostoru parametara, da se ponaša kao desktop učitelj na
desktop promptovima i kao mobilni učitelj na mobilnim promptovima. Pošto je
uputstvo *on-policy* (učenik generiše, pa se poredi sa učiteljem na sopstvenim
uzorcima) i uslovljeno platformom, nema distribucijskog mismatch-a i nema
prosečenja — signal svake platforme ostaje čisto odvojen.

## Kako radi (intuicija)

### Faza 1 — napravi dva eksperta

SFT-uj Qwen3-VL-32B-Thinking odvojeno na Uni-GUI desktop trajektorijama i na
Uni-GUI mobilnim trajektorijama. Sad imaš π_ref^d i π_ref^m — dva učitelja,
svaki specijalista.

### Faza 2 — destiluj u jednog učenika, onlajn

Učenik je Qwen3-VL-8B-Thinking. Svaki korak treniranja:

1. **Rollout.** Uzmi mešoviti batch desktop + mobilnih promptova. Za svaki
   prompt učenik generiše G rollouta (papir koristi G=8).

2. **Nagrada.** Svaki rollout dobija strukturisanu ishodnu nagradu (Jed. 8):
   +1.0 ako akcija potpuno matchuje target po svim dimenzijama (tip akcije,
   koordinata-u-bbox, smer skrola, match tastera/teksta), −0.5 ako je delimično
   važeća, −1.0 ako je neparsabilna/nevažeća. Ovo je *rule-based* nagrada, ne
   naučeni model.

3. **Grupna prednost.** Unutar grupe od G rollouta po promptu, računa se
   standardni GRPO baseline: A_t = R(x,y) − mean(R preko grupe). Tokeni u
   rolloutu koji nadmaše braću dobijaju pozitivnu prednost.

4. **Uslovljeni učiteljev KL po platformi (srž MOPD-a).** Za svaki rollout,
   pogledaj njegovu platformu i uzmi log-probabilnosti *odgovarajućeg*
   učitelja preko tokena tog rollouta. Računaj K3 KL estimator (Jed. 4–5):

   δ_t = log π_ref(y_t | h_t) − log π_θ(y_t | h_t), clamp-ovan
   ρ_t = exp(δ_t)
   D̂_KL = ρ_t − δ_t − 1

   Ovaj estimator je **ne-negativan** (ρ − δ − 1 ≥ 0 uvek, jer je e^δ ≥ 1 + δ)
   i **nepristrasan** za KL(π_θ || π_ref) pod uzorcima izvučenim iz π_θ, sa
   nižom varijansom od sirovog log-ratio-a. Ta ne-negativnost je bitna:
   znači da KL član može samo da *vuče učenika ka učitelju*, nikada da ga
   proizvoljno gura od njega, što stabilizuje treniranje.

5. **Adaptivna KL maska.** Evo suptilnog ali važnog trika. Za svaku grupu
   promptova, ako srednja nagrada grupe već prelazi prag τ_KL, postavi KL
   težinu μ=0 za tu grupu. Prevod: *ako učenik ovde već uspeva, pusti ga da
   istražuje — ne vuci ga nazad ka učitelju.* KL ograničenje se uključuje samo
   gde učenik *podlaže*, što je tačno tamo gde je učiteljevo uputstvo
   najkorisnije.

6. **Clipped cilj.** Kombinuj sve u PPO-stil cilj (Jed. 10–12):

   L = L_PPO(clip ratio, prednosti) + β · L_MOPD(K3 KL, maskirano sa μ)

   sa asimetričnim klipingom (ε_low=0.2, ε_high=0.28) i malim β=0.01.

U inferensu radi samo učenik — nema učitelja, nema rutiranja. Ponašanje
specifično za platformu je upakovano u jedan skup težina.

## Šta sam naučio implementirajući

**K3 estimator je neveličani heroj.** Implementirao sam `D̂ = exp(δ) − δ − 1`
direktno i ne-negativnost je upečatljiva: matematički je nemoguće da član ode
u negativno (to je e^δ vs. njegova tangenta u δ=0). To znači da je KL gradijent
uvek "vuci ka učitelju," nikad "beži od učitelja proizvoljno." Sa sirovim
reverse-KL ili log-ratio možeš dobiti lude negativne vrednosti koje
destabilizuju politiku. K3 to zaobilazi u potpunosti. Clamp-ovanje δ je
jedina numerička zaštita koja treba.

**Adaptivna maska je varljivo prosta ali posledična.** To je jedna poređenje
(srednja nagrada grupe > τ_KL → μ=0). Ali efekat je da učitelj deluje kao
*mreža za bezbednost*, ne kao povodac. U mom toy run-u, čim je nagrada prešla
τ_KL=0.5, KL član je išao tačno na 0.0000 za te grupe i učenik je optimizovao
čist PPO — tačno kao što treba. Bez toga, prejako učiteljevo ograničenje bi
kapiralo učenikovu performansu na učiteljevu, sprečavajući svako istraživanje
izvan nje.

**Rutiranje platforme je samo operacija indeksiranja.** Očekivao sam nešto
fensi, ali Jed. 7 je bukvalno `teacher_lp = where(platform==mobile,
mobile_teacher_lp, desktop_teacher_lp)`. "Multi-teacher" okvir je konceptualno
bogat, ali mehanički trivijalan: maska po rolloutu bira koje smrznute
učiteljeve logite koristiti. Ovo metodu čini trivijalno proširljivom na N
platformi — samo N učitelja i N-way rutirajuća maska.

**Mixed-SFT kao baseline je pravi kontrast.** U mojoj toy implementaciji
mixed-SFT učenik je već pogodio 100% jer su sintetički šabloni razdvojeni po
ključnoj reči zadatka, pa MOPD nije imao šta da doda. U *pravom* papiru
mixed-SFT i model-merging oba padaju na balansiranju platformi — što je cela
poenta. Toy ne može da reproducuje katastrofalno-zaboravljanje jer je
sintetički zadatak previše lak, ali čisto vežba svaku jednačinu. Ovo iskreno
beležim u README-u.

## Šta me iznenadilo / bilo teže nego očekivano

**8B učenik pobeđuje 32B osnovni model.** UI-MOPD-ov 8B učenik dobija 12.0%
na MobileWorld naspram 9.4% za 32B *osnovni* Qwen3-VL-Thinking. To nije razmera
— to je znanje ponašanja specifično za platformu preneto destilacijom. Ovo je
najčistiji dokaz da dobici dolaze od *metode*, ne od bacanja parametara na
problem.

**TIES merging je iznenađujuće destruktivan.** TIES merging — navodno
principijelna metoda proseka težina — obara AndroidControl grounding sa
78.73% na 74.01%, dok UI-MOPD ga *poboljšava* na 80.05%. Lekcija: statično
prosečnje *parametara* dva specijaliste je fundamentalno drugačije od
destilacije njihovog *ponašanja* on-policy. Parametri žive u ne-euklidskom
prostoru gde prosečenje može otkazati specijalizovane pravce; bihejvioralna
destilacija poštuje geometriju jer radi u stvarnoj distribuciji izlaza
politike.

**Anomalija koju ne mogu da objasnim.** Na OSWorld-G Text Matching, TIES Merge
dobija 47.37% — *više* i od osnovnog (31.58%) i od UI-MOPD (42.11%) — uprkos
tome što je TIES gori od oba na bitno svakom drugom sub-metriku. Ova jedna
inconsistencija sedi čudno u inače koherentnoj slici i ne diskutuje se u
papiru. Možda je šum (MobileWorld ima samo 117 zadataka; OSWorld-G Text
Matching ima ~40 uzoraka) ili pravi kvirk kako TIEWS premešta tekst-match
featurove. U svakom slučaju, to je stvarna bradavica.

**"Multi-platform" tvrdnja je zapravo "dual-platform."** Samo desktop (OSWorld)
i mobilni (MobileWorld) se vrednuju — nema web GUI agenata. Zvati to
"multi-platform continual learning" je tehnički pošteno (2 ≥ 2) ali
generalnost metode preko dve platforme se tvrdi, ne demonstrira.

## Reference

- Papir: https://arxiv.org/abs/2607.04425
- Projektna stranica: https://elispectre.github.io/UI-MOPD/
- Breakdown: `breakdown.md`
- Moja implementacija: `implementation/` (Jed. 1–12, toy razmera)
