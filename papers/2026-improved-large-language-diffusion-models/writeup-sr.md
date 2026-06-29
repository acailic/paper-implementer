# Writeup — iLLaDA: Improved Large Language Diffusion Models

> Evo kako bih ovo objasnio prijatelju pre piva, ako me pita "šta si čitao?"

> **Jezici:** [English](writeup.md) · Srpski (ovaj fajl)

---

Priča u jednoj rečenici: ljudi koji su napravili LLaDA — prvi ozbiljan pokušaj da se napravi veliki jezički model koji generiše tekst koristeći difuziju umesto autoregresivnog generisanja token po token — vraćaju se sa većim, boljim modelom koji je sada zaista konkurentan modelima kao što je Qwen2.5 7B.

Evo konteksta. Svaki veći jezički model koji si čuo (GPT-4, Claude, Llama, Qwen) radi na isti način: nahraniš mu prompt, i on generiše jedan token po jedan, slijeva nadesno, gdje svaki novi token ovisi o svim prethodnim. To je autoregresivna generacija. Dominantna je godinama i s razlogom — dobro radi.

LLaDA (sa NeurIPS 2025, oral papir) postavio je drugačije pitanje: šta ako treniraš jezički model sa difuzijom umesto toga? Ideja je bliža onome kako difuzija radi za slike — počneš sa nizom mask-tokena (zamisli kao čist šum), i kroz više koraka model refinira sve pozicije istovremeno dok ne dobiješ čist tekst. Potpuno dvosmjerna pažnja, nema ograničenja slijeva nadesno.

Originalni LLaDA je pokazao da ovo može da radi na 8B parametara, ali je bio treniran na samo 2.3 biliona tokena i znatno je zaostajao za Qwen2.5 7B na benchmarkovima. Zanimljiv proof of concept, ali još ne i konkurentan.

## Šta iLLaDA radi drugačije

iLLaDA ne mijenja fundamentalnu ideju. I dalje koristi masked difuziju sa potpuno dvosmjernom pažnjom. Što se mijenja je inženjering i skaliranje.

**Pet puta više trening podataka.** LLaDA je bio treniran na 2.3T tokena. iLLaDA dobija 12T. To je masivan scale-up i najveći pojedinačni doprinos poboljšanju.

**Arhitektonske ispravke.** Prebačeno sa standardnog multi-head attention-a na grouped-query attention (GQA) — 32 query heada dijeli 8 key/value head grupa. Ovo smanjuje KV-cache memorijski otisak, što je bitno jer je nedavni rad pokazao da difuzni LLM-ovi mogu koristiti KV-cache stil inferencije. Takođe su vezali input embedinge i output LM head zajedno (manje parametara, isti kapacitet). Povećali FFN dimenziju sa 12,288 na 14,336. Udvostručili kontekst sa 4,096 na 8,192.

**Iznenadujuće iskreno ispravljanje LR rasporeda.** Počeli su trening sa konstantnom stopom učenja (nakon warmup-a). U jednom trenutku loss je prestao da opada. Zato su prebacili na cosine decay. Loss je počeo ponovo da opada. Ovo je osvježavajuće nepretenciozno — nema nemačke nove schedule, samo "loss je zastao pa smo ga promijenili."

**Ujedinjeni SFT format.** Prethodno difuzno fino podešavanje LLM-a držalo je prompt vidljiv i maskiralo samo dio sa odgovorom. iLLaDA kaže: ne, maskiraj sve. Prompt, odgovor, čak i EOS token. Istim formatom kao pre-treniranje. Ovo izbjegava train-inference mismatch i prirodno omogućava generisanje promjenjive dužine.

**12 epoha SFT-a.** To je mnogo. Većina autoregresivnih modela radi 1-3 epohe supervised fine-tuniranja. Ali difuzni modeli očito nastavljaju da se poboljšavaju na ponovljenim podacima — svojstvo koje je uočeno i drugdje (Ni et al. 2025 su pokazali da su difuzni modeli "super učenici podataka" koji profitiraju od treninga na istim tokenima mnogo puta). Ablacija potvrđuje: GSM8K, MATH i MMLU-Pro svi se poboljšavaju od epohe 3 do 12.

**Confidence-based scoring za multiple-choice.** Ovo je pametno. Za benchmarkove koji nude izbore (kao HellaSwag ili ARC), treba ocijeniti svakog kandidata i odabrati najboljeg. Direktan pristup je da izračunaš procjenu likelihood-a. iLLaDA umjesto toga koristi "confidence score" — počni od all-masked kandidata, i na svakom koraku otkrij onaj token za koji je model najsigurniji, akumulirajući log-probabilitete. Nije pravi likelihood, jeste heuristika, ali radi bolje: +1.3 na PIQA, +0.6 na ARC-Challenge, +2.3 na HellaSwag.

## Gdje stvari stoje

Rezultati baznog modela su zaista impresivni. iLLaDA 8B pobjeđuje Qwen2.5 7B u prosjeku na 8 benchmarkova (63.9 vs 63.3). Specifično pobjeđuje na MMLU-u (69.5 vs 73.3 — i dalje iza), BBH (71.9 vs 78.9 — iza), ARC-Challenge (74.8 vs 77.2 — blizu), i GSM8K (81.9 vs 79.0 — zapravo ispred). BBH razmak je još uvijek veliki (+7), ali ARC-C i GSM8K su blizu.

U poređenju sa originalnim LLaDA-om, napredak je dramatičan: 51.1 → 63.9 u prosjeku. BBH ide sa 39.6 na 71.9 (+32.3!). ARC-Challenge sa 49.7 na 74.8 (+25.1). Ovo nisu margine, ovo su prave razlike.

Instruct model priča drugu priču. iLLaDA-Instruct (67.1 u prosjeku) je i dalje daleko iza Qwen2.5 7B Instruct (77.1 u prosjeku). To je razmak od 10 poena. Autori su direktan o razlogu: Qwen2.5 koristi reinforcement learning alignment nakon SFT-a, a iLLaDA ne. Nkoliko RL metoda za difuzne LLM-ove već postoji (VRPO, diffu-GRPO, MDPO, ESPO), pa je primjena očigledan slijedeći korak.

Jedna zanimljiva anomalija: iLLaDA-Instruct pobjeđuje Qwen2.5 na GSM8K (89.0 vs 88.0). Mala razlika, ali simbolična.

## Šta me najviše zanimalo

**Svojstvo "super učenika podataka" je stvarno i praktično.** 12 epoha na 25B tokena SFT podataka, i dalje se poboljšava. Za autoregresivne modele, trening na istim podacima 12 epoha bi vjerovatno prouzrokovao overfitting. Difuzni modeli očito drukčije rukuju ponovljenim podacima. Ovo ima prave implikacije za svakog ko radi sa ograničenim instruct podacima — difuzni modeli ti daju više kilometraže od istih podataka.

**Problem ponavljajućih razmišljačkih petlji je specifičan za difuziju i malo smješan.** Na teškim instruct problemima, iLLaDA se ponekad zarobi u petlju: "Sačekaj, da provjerim ponovo. Zapravo, sačekaj. Da razmislim ponovo. Hmm, da razmislim pažljivije..." beskonačno. Njihov fix je da postepeno povećavaju vjerovatnoću emitovanja stop-thinking tokena kako generacija duže traje. Pragmatično, ali zvuči kao flaster na dublji problem o tome kako dvosmjerni modeli rukuju višestepenim rasuđivanjem.

**Confidence scoring je obmanjujuće jednostavan.** Nije teorijski utemeljen procjenitelj likelihood-a. Samo je: "na svakom koraku, za koji token je model najsigurniji? Ocijeni tog." I pobjeđuje teorijski motiviranu gornju granicu. Jednostavne heuristike pobjeđuju teoriju u praksi — klasika.

## Šta nedostaje

Papir je kratak (10 stranica uključujući appendix). Nema ablacije za pojedinačne arhitektonske promjene — ne znamo da li GQA, vezani embedingi, ili veći FFN zapravo znače nezavisno. Ne znamo trening compute budžet (nema FLOPs-a). Ne znamo sastav 12T korpusa. Nemamo scaling krive izvan 8B.

To je u redu za ono što je papir — inženjerski izvještaj "uvećali smo i radi." Ali znači da ne možeš izvući generalizabilne principle iz njega dobro. Da li je bitno 12T tokena, ili GQA, ili SFT format, ili LR schedule promjena? Vjerovatno sve malo, ali papir ne pomaže da to razdvojiš.

## Verdikt

iLLaDA je solidan inženjerski nastavak koji difuzne jezičke modele čini zaista konkurentnim autoregresivnim na 8B skali — bar za bazne modele. Glavna poruka je jasna: difuzni paradigma radi, skalira se, i sa pravim inženjeringom možeš parirati Qwen2.5 7B. Instruct razmak je stvaran ali rješiv sa RL.

Nije papir koji mijenja paradigmu. Ideje su inkrementalne. Ali rezultat je bitan: dodatna validacija da ne trebaš autoregresivnu faktorizaciju da bi izgradio snažan jezički model. Da li difuzija može da se takmiči na 70B+ i da li RL alignment zatvara instruct razmak su dva velika otvorena pitanja.

## Reference
- Papir: https://arxiv.org/abs/2606.25331
- Kod & težine: https://github.com/ML-GSAI/LLaDA
- Prethodnik (LLaDA, NeurIPS 2025 Oral): https://arxiv.org/abs/2502.09992
- Breakdown: `breakdown.md`
