# DanceOPD: On-Policy Generative Field Distillation — Prikaz na bosanskom

## Uvod

Problem je jednostavan na papiru, težak u praksi. Treba ti jedan model koji i generiše slike iz teksta i edituje postojeće slike — lokalno i globalno. Ali te sposobnosti se međusobno bore. Kada model nauči editovanje, kvalitet T2I generisanja pada. Kada pokušaš sjediniti lokalno i globalno editovanje, rezultat je kompromis koji ne zadovoljava ni jedno ni drugo. ByteDance Seed i NUS tim predlaže DanceOPD — pristup koji razmišlja o ovom problemu na potpuno nov način.

## Polje vidjenja umjesto spajanja težina

Umjesto da miješa težine različitih modela ili kombinuje podatke iz raznih zadataka, DanceOPD posmatra svaki model kao brzinu polje (velocity field) nad istim latentnim prostorom. Model za T2I generisanje definiše jedno polje, model za editovanje drugo, model za stil treće. Kompozicija više sposobnosti tada postaje pitanje kako student treba da upita ova polja — koje polje, gdje, i koliko puta po trajektoriji.

## Tri problema, tri rješenja

**1. Ambiguum ciljnog polja.** Ako prosečiš brzine T2I i edit modela, dobiješ smjer koji ne odgovara nijednom pravom zadatku. Rješenje: svaki uzorak ide tačno u jedan ledeni model. T2I uzorak upita T2I polje, edit uzorak upita edit polje. Kompozicija se dešava statistički, kroz mnoge optimizacijske korake.

**2. Nepoklapanje distribucije stanja.** Ako teacher polje evaluiraš na fiksnim offline stanjima, student ostaje nenadgledan na stanjima koja zaista posjećuje pri generisanju. Rješenje: izvrši studentov rollout (16 Euler ODE koraka) i upita teacher na studentovim vlastitim stanjima. To je "on-policy" dio — teacher nadgleda tamo gdje student zaista jeste.

**3. Korelacija trajektorije.** Prirodno je reći "upitaj više stanja po rollout-u za gušću superviziju". Ali ta stanja dijele isti šum, isti prompt, istu istoriju puta — nisu nezavisna. Rješenje: samo jedan upit po uzorku, postavljen na nisko-šumnoj (semantičkoj) strani trajektorije gdje su informacije specifične za zadatak najkoncentrisanije.

Gubitak je elementaran: `||v_student - v_teacher||²` — običan MSE na rutiranom, on-policy upitu.

## Rezultati

Četiri eksperimentalna okruženja na Z-Image modelu:

| Okruženje | Ključni rezultat | Dobitak |
|-----------|------------------|---------|
| T2I + Edit | GEditBench | +8.1% naspram najboljeg OPD baseline-a |
| Lokalno + Globalno Edit | GEditBench | +16.1% naspram najbolje konkurencije |
| Apsorpcija realizma | Realism reward | +9.9% naspram off-policy, zatvara 85.3% jaz |
| CFG apsorpcija | GEditBench | +7.6% naspram train-only apsorpcije |

## Dijagnostika: zašto svaki izbor ima smisla

Ablacije su izuzetno temeljite i svaki dizajnerski izbor je dobro potkrepljen:

- **Hard routing vs. soft miješanje:** +15.2% pod MSE-om. Problem je u konstrukciji cilja, ne u funkciji gubitka.
- **Nisko-t vs. drugi timestep-ovi:** +23.7% naspram median-t, +19.5% naspram high-t. Informacije vezane za zadatak zaista su koncentrisane u nisko-šumnim stanjima.
- **Jedan upit vs. gusti upiti:** K=1 pobjeđuje K=2,4,8,16 za 7.9–16.6%. Korelirana stanja nisu besplatna supervizija.
- **SDE dekorelacija:** Stohastički rollout oporavlja 18.4% gustog degradiranja ali ostaje 8.6% ispod K=1. Potvrđuje da je korelacija problem, ali ne daje bolji default.
- **Običan MSE vs. alternative:** Timestep-weighted, KL, DMD-EMA, consistency, feature distillation — svi su gori. Kad je cilj determinističko polje brzine, direktna regresija je pravi alat.
- **Inicijalizacija:** Local-edit init pobjeđuje merged za 37.2%. Kvalitet inicijalnog rollout-a je bitan jer teacher polja tamo upituju rano u treningu.

## Efikasnost

DanceOPD koristi 16-koračni rollout po koraku treninga (kao svaki on-policy metod) ali evaluira samo jedno stanje sa gradijentom (K=1). DiffusionOPD evaluira svih 16, a Flow-OPD dodaje PPO overhead i 2× mikro-batch faktor. DanceOPD je brži od oba, a daje bolje rezultate.

## Ograničenja

Svi teacher modeli moraju raditi na istom latentnom prostoru sa kompatibilnom parametrizacijom — ne možeš trivialno komponovati modele iz različitih arhitektura. Rutiranje je unaprijed definirano identitetom podatka, pa ne radi dobro za prompove koji istovremeno trebaju više sposobnosti. Evaluacija realizma koristi proprietarni reward model, što ograničava reproduktivnost tog eksperimenta.

## Zaključak

DanceOPD je dobro motiviran, pažljivo validiran i praktično koristan rad. Polje-bazirano gledište prirodno se uklapa u flow-matching modele, a tri dizajnerska izbora svaki adresira stvarni problem sa jasnim ablacijskim dokazima. Jednostavnost konačnog objektiva (običan MSE) je jačina — sugerira da su autori prepoznali pravi problem umjesto da ga prekompleksuiraju. Za svakoga koji gradi višesposobne generativne sisteme, ovaj rad je vrijedan pažnje.
