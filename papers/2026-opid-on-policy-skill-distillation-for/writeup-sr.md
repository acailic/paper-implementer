# OPID: On-Policy Skill Distillation za Agentic Reinforcement Learning

> Objašnjenje rada u svom stilu, na našem.

## U jednoj rečenici

OPID je okvir za trening agentic LLM modela koji iz završenih trajektorija
izvlači "vještine iz vidika u prošlost" (hindsight skills) i koristi ih kao
dodatan, token-po-token signal za učenje — uz standardni GRPO outcome reward,
ali bez ikakvog dodatnog kompleksa na inference.

## Problem

Kada treniraš LLM agenta za višekoračne zadatke — manipulacija objektima u
kući, kupovina na webu, traženje informacija pretraživačem — nagrada je
uglavnom binarna i kasni. Čitav niz koraka završi, pa onda kaže: uspjeh ili
neuspjeh. Nema povratne informacije koji je tačno korak bio pogrešan. GRPO
normalizuje nagrade unutar grupa, ali svaki token u trajektoriji dobija isti
skalarni advantage. Agent ne zna da li je greška bila na koraku 3 ili koraku
17.

Prethodni radovi su pokušali popraviti ovo sa skill-conditioned distillation —
davanjem modelu dodatnog naturálnog-jezičnog guidancea tokom treninga. Ali ti
metodi trebaju eksterne biblioteke vještina koje netko mora održavati, a
pronađene vještine mogu biti zastarjele ili neodgovarajuće za ono što
trenutna politika modela zapravo susreće.

## Ideja

Autori primjećuju nešto jednostavno i elegantno: agentove vlastite završene
trajektorije već sadrže sve potrebno znanje o donošenju odluka. Uspješna
trajektorija pokazuje valjan workflow; neuspješna pokazuje šta treba izbjegavati.
Treba samo izvući to znanje kao vještine i koristiti ga kao trening signal.
Pošto te vještine dolaze iz vlastitih rollout-a trenutne politike,
garantovano odgovaraju distribuciji stanja — nema mogućnosti mismatch-a.

## Kako radi

Zamisli analizu utakmice nakon meča. Kad agent završi zadatak, analizator
(eksterni LLM) ga "sjedne" i pita: "Kakva je bila ukupna strategija? I na
kojim ključnim momentima je stvari pošlo dobro ili loše?" Analizator proizvodi
dvije vrste vještina:

1. **Epizodna vještina**: Velika slika. Za uspjeh je nešto poput "Prvo pronađi
   objekat, onda ga očisti na sudoperu, onda ga smjesti." Za neuspjeh:
   "Pokušavao si staviti prljav objekat u korpu bez čišćenja."

2. **Step-level vještine**: Precizno guidance na 2-5 kritičnih koraka.
   Npr. "Na koraku 0, idi direktno do countertop-a gdje je kotao" ili
   "Na koraku 2, provjeri da li sapunica treba čišćenje."

OPID zatim na svakom koraku bira *najodgovarajuću* vještinu — step-level na
kritičnim momentima, epizodnu svugdje drugdje. To je critical-first routing.

Da bi pretvorio ove vještine u signal za učenje, OPID ubaci odabranu
vještinu u interakcijsku historiju i zamoli politiku da ponovo ocijeni svoj
odgovor — sa i bez vještine. Ako skill-augmented kontekst čini neki token
vjerovatnijim, taj token dobija pozitivan advantage. Ako manje vjerovatnim —
negativan.

Ovaj token-level advantage se dodaje standardnom GRPO trajektorija-level
advantage-u. Epizodni advantage kaže politici "ova trajektorija je bila
dobra/loša ukupno", dok skill advantage kaže "na ovom specifičnom tokenu,
evo šta retrospekcija kaže da si trebao napraviti."

Pošto skill signal dolazi iz uparenog scoring pass-a (ista politika, isti
odgovor, različiti kontekst), garantovano je on-policy. I na inference-u
nije potrebno ništa od toga — politika je internalizirala znanje o vještinama.

## Što sam naučio

Najvažnija implementacijska stvar je prompt za analizator. Paper koristi
pažljivo strukturirani JSON prompt sa tri polja: episode_summary,
episode_skill i step_skills (dictionary koraka do vještina). Indeksi koraka
su 0-bazirani i moraju se poklapati sa trajektorijom. Dobro ispisati ovaj
prompt je vjerovatno pola posla.

Upareni scoring je takođe ne-trivijalan. Treba napraviti pun forward pass
kroz politiku dva puta po koraku trajektorije — jednom sa originalnim
kontekstom, jednom sa skill-augmented kontekstom. Ovo udvostručuje compute
po koraku, ali samo tokom treninga. Stari parametri politike su zamrznuti.

λ_skill = 0.001 je iznenadujuće mali. Skill advantage je nježan nudge,
ne dominantna sila. Ima smisla — želiš da RL outcome signal bude primaran,
sa vještinama koje pružaju fine-grained shaping.

## Što me iznenadilo

Najveće iznenađenje je koliko OPID nadmašuje Skill-GRPO. Skill-GRPO koristi
eksterne vještine tokom treninga ali ih uklanja na inference, i rezultat je
često *gori* od čistog GRPO. Na ALFWorld Qwen2.5-3B, Skill-GRPO bez inference
vještina daje 60.2 dok čisti GRPO daje 75.0. Train-test mismatch je
razarajući. OPID-ov pristup destilacije vještina u parametre modela u potpunosti
izbjegava ovaj problem.

Rezultat efikasnosti po uzorku je značajan: OPID sa 60% podataka je na nivou
GRPO sa 100% podataka. Ovo sugeriše da dense token-level supervision izvlači
značajno više signala za učenje po rollout-u nego rijetke outcome nagrađivanje.

## Refrence

- Papir: https://arxiv.org/abs/2606.26790
- Kod: https://github.com/jinyangwu/OPID
- Breakdown: `breakdown.md`
