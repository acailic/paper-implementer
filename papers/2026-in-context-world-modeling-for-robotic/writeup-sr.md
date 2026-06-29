# In-Context World Modeling za Robotsku Kontrolu — Writeup

> 🇬🇧 English version: [writeup.md](writeup.md)

**ArXiv:** 2606.26025v2 · Juni 2026
**Autori:** Siyin Wang, Junhao Shi, Senyu Fei, Zhaoyang Fu, Li Ji, Jingjing Gong, Xipeng Qiu

---

## Verzija u jednom pasusu

VLA modeli (poput π0, OpenVLA, RT-2) pucaju kada promijeniš kut kamere jer su tokom treninga utisnuli konfiguraciju sistema u svoje težine. Rješenje iz ovog rada je jednostavno kao i genijalno: prije nego što robot obavi zadatak, pustiš ga da se malo "njamori" nasumice nekoliko sekundi, snimiš šta se desi, i takaš te isječke kao kontekst modelu. Tokom treninga, slični nasumični isječci se dodaju na početak svakog trening uzorka, pa model uči da ih koristi kao kalibracioni signal. Na testiranju, model implicitno razaznae kut kamere i mapiranje akcija iz ovih isječaka — nema fine-tuninga, nema demonstracija, nema dodatnih parametara. I radi: +13% na nepoznatim kamerama u simulaciji i ogromna poboljšanja na pravom UR5e robotu.

---

## Problem

Zamisli da kontrolišeš robota džojstikom, ali nikad prije nisi koristio baš ovaj setup. Gurneš naprijed — da li robot ide lijevo, desno ili naprijed? Nećeš odmah pokušati zadatak. Malo ćeš pokretati džojstik, gledati šta se dešava. Za par sekundi gradiš interni model — shvatiš kako inputi korespondiraju outputima. Tek onda izvršavaš zadatak sa povjerenjem.

VLA modeli to ne umiju. Oni uče π(a | o, l) — akciju na osnovu opservacije i jezika. Kut kamere, morfologija robota, offseti montiranja — sve je to implicitno fiksirano u trening podacima i upijeno u težine. Postaviš drugu kameru? Model nema mehanizam da shvati novu vezu između opservacije i akcije. Prosto generiše pogrešne akcije.

Standardni fix je fine-tuning za svaki novi setup. Skupo je, zahtijeva ljudsku intervenciju, i ne skalira se ka viziji "generalist robota" koju svi jure.

## Ideja

Ovaj rad reframira problem kao problem identifikacije sistema. Policija treba da zna ψ (konfiguraciju sistema) u momentu testiranja. Umjesto fine-tuninga, papir predlaže da se ψ povrati iz kratke historije interakcija.

Trik: iskoristi kontekstni prozor transformera. Većina ICL rada u robotici koristi kontekst za specifikaciju ponašanja — "evo ti demo, kopiraj." ICWM koristi kontekst za identifikaciju sistema — "evo ti nekoliko nasumičnih pokreta, shvati kako sistem radi."

Tokom treninga, svaki uzorak dobije N=5 nasumičnih isječaka interakcije na početak. Ti isječci dolaze iz različitih konfiguracija u trening podacima. Model je primoran da nauči izvlačiti konfiguracione informacije iz konteksta da bi pravilno predvidio akcije.

Na testiranju, robot napravi ~20 nasumičnih pokreta (traje oko 5-6 sekundi), uzmeš 5 isječaka od toga i staviš na početak konteksta. Jedan forward pass. Nema ažuriranja gradijenata. Nisu potrebne demonstracije za taj zadatak.

## Kako radi (Intuicija)

Zamisli ovako: jedna fotografija radnog prostora robota ne govori ti kut kamere — ista scena izgleda slično iz mnogih uglova. Ali *video* robota koji se pomjera, koji pokazuje kako objekti klize po slici dok se end-effector pomjera, otkiva geometriju kamere. Veza "robot se pomjerio lijevo" i "objekti su se pomakli desno na slici" enkodira tačku gledišta.

Model vidi 5 takvih isječaka prije zadatka. Svaki isječak je tripleta: početna slika → akcija → krajnja slika. Iz njih implicitno rekonstruiše kameru-do-robota mapiranje. Kad onda vidi zadatak, može pravilno interpretirati prostorne odnose i generisati tačne akcije.

Formalni argument (Propozicija 1) je elegantan: pod blagim pretpostavkama, sekvenca parova (opservacija, akcija) nosi strogo više informacija o ψ nego bilo koja pojedinačna opservacija. To važi za *bilo koju* distribuciju akcija — pa radi i sa nasumičnim pokretima. Ne treba ti zadatak-specifična eksploracija.

Cijela stvar dodaje **nula dodatnih parametara**. Interakcijski kontekst procesira isti Qwen2.5-VL backbone. Jedina promjena je format trening podataka i 5-6 sekundi kalibracionog "njamoranja" prije deploymenta.

## Šta me iznenadilo

**Pogrešan kontekst je gore od bez konteksta.** Kad hrane modelu isječke iz ugla od 180° razlike, performanse padaju ispod baselina bez konteksta. Model ne profitira samo od dužih sekvenci ili dodatnih tokena — zaista ekstraktuje konfiguracione informacije, i pogrešne informacije ga aktivno zbunjuju. Simetrična magnituda (dobitak od pravog konteksta ≈ gubitak od pogrešnog) je čista demonstracija.

**Strategija sondiranja gotovo ne igra ulogu.** Nasumično, samo-XY, samo-Z, samo-rotacija — sve poboljšava baselin za slične margine. Ne treba pametna strategija. Bilo kakvi prostorno raznoliki poketi otkrivaju dovoljno dinamike manifold-a. To su odlične vijesti za prave deploymente: samo malo pomiči i kreni.

**In-context trening je esencijalan.** Uzmi standardni BC model (treniran bez konteksta) i pokušaj dodati interakcijske isječke na testiranju. Performanse kolapsiraju ispod 1%. Kapacitet ne nastaje samo iz sekvencijalnog modelovanja — moraš eksplicitno trenirati model da koristi kontekst za inferenciju konfiguracije.

**Najveći dobici su na dugim zadacima.** Na LIBERO-Long, ICWM pobjeđuje multi-view baselin za 26.3% na nepoznatim kamerama. Papir objašnjava: dugi zadaci amplificiraju male prostorne greške iz pomaka kamere, uzrokujući kaskadne neuspjehe. Kalibracija putem ICWM-a sprečava inicijalnu akumulaciju greške. Ima intuitivnog smisla — mala greška u dubini kod jednog pokreta je sanjiva; kod 10-koračne manipulacije, seomete pogromi.

**Nema koda.** Za rad sa tako čistom idejom, to je šteta. Volio bih vidjeti reprodukciju.

---

## Reference

- Rad: [arXiv 2606.26025](https://arxiv.org/abs/2606.26025)
- LIBERO Benchmark: [Liu et al., NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/8c3c668620ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)
- Qwen2.5-VL: [Bai et al., 2025](https://arxiv.org/abs/2502.13923)
- FAST Action Tokenizer: [Pertsch et al., 2025](https://arxiv.org/abs/2501.09747)
- π0: [Black et al., 2024](https://arxiv.org/abs/2410.24164)
- OpenVLA: [Kim et al., 2024](https://arxiv.org/abs/2406.09246)
- RT-2: [Zitkovich et al., CoRL 2023](https://proceedings.mlr.press/v229/zitkovich23a.html)
- ICRT (in-context imitation learning): [Fu et al., ICRA 2025](https://arxiv.org/abs/2406.09246)
