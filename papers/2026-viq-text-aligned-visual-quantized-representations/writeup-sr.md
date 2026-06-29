# Writeup — ViQ: Text-Aligned Visual Quantized Representations at Any Resolution

> Evo kako bih ovo objasnio prijatelju pre piva, ako me pita "šta si čitao?"

> **Jezici:** [English](writeup.md) · Srpski (ovaj fajl)

---

Jednostavna priča ide otprilike ovako.

Zamisli da gradiš multimodalni chatbot. Treba da učitaš slike u jezički model.
Trenutno svi koriste vizuelni enkoder kao CLIP ili SigLIP koji ispljunje dug
niz float vektora. Ti vektori su kontinualni — svaki broj može biti bilo šta
od -1.7 do 0.3 do štogod. Jezički model, međutim, radi isključivo sa diskretnim
tokenima — cele reči, subreči, celi brojevi iz nekog rečnika.

To je mismatch. I skupo je. Svaki trening korak moraš da pokreneš ovaj veliki
vizuelni enkoder, provučeš sve te float vektore kroz LLM attention, i platiš
za to u GPU satima.

Šta bi bilo da možeš jednostavno da... tokenizuješ slike isto kao što
tokenizuješ tekst? Pretvoriš sliku u niz integera, kao reči u rečenici, i
nahranjuješ ih direktno u LLM? To pokušavaju quantized vizuelni enkoderi.

Problem je da postojeći pokušaji propadaju na jednoj od dve stvari. Ili
integer kodovi čuvaju vizuelne detalje (možeš pristojno da rekonstruišeš sliku)
ali gube sav semantički značaj (LLM ne može da odgovori na pitanja o njoj), ili
čuvaju značaj ali unište detalje. Izaberi jedno.

ViQ kaže: oba. Evo kako.

## Trik sa dve faze

Ključni uvid je da ne možeš samo da nalepiš quantization sloj na pretrenirani
vizuelni enkoder i to je to. Ako ideš direktno iz 1536-dimenzionalnog
kontinualnog prostora u 6-dimenzionalni diskretni, izgubiš previše. Papir
pokazuje to brutalno: direktna kvantizacija pada prosečan skor sa ~69 na 61.
To je katastrofa.

Zato ViQ radi to u dve faze.

**Faza 1** je o tome da vizuelni enkoder postane "multimodalan-svestan."
Uzmeš SigLIP2 enkoder, zameniš fiksne pozicionalne embedinge za resizable
da može da radi sa bilo kojom rezolucijom, pa ga treniraš sa jezičkom
supervizijom — daješ mu tripletove slika-tekst-odgovor i koristiš cross-entropy
loss kroz mali LLM. Dok to radiš, držiš zamrznutu kopiju originalnog
enkodera kao učitelja koji se pobrine da učenik ne zaboravi originalno
vizuelno znanje (self-distillation preko kosinus sličnosti na class tokenu).

Ova faza sama po sebi daje bolji kontinualni vizuelni enkoder. Ali još uvek
je kontinualan.

**Faza 2** je gde se magija dešava, i ima dve pod-faze.

**Faza 2-1: proksimalna reprezentacija.** Pre kvantizacije, dodaš bottleneck
koji komprimuje 1536 dimenzija na 128, i onda primeniš L∞ norm — prisiliš
svaki feature da živi na površini hiperkocke, gde je svaka dimenzija ograničena
na [-1, 1]. To je "meko slijetanje" pre tvrdog slijetanja kvantizacije.

Zašto baš L∞? Ablacija priča celu priču. Bez regularizacije: 60.9. L2:
67.9. L∞: 68.7. L∞ norm radi bolje jer ograničava prostor uniformnije — sve
dimenzije su vezane za isti opseg, pa su features ravnomerno raspoređeni i bliže
kvantizacionim sidrima.

U ovoj fazi dodaš i rekonstrukcijski branch. Ali ovde je pametan dio:
umesto da rekonstruišeš piksele (što treba GAN losse, perceptual losse, i
je skupo), predviđaš latentnu reprezentaciju *pretreniranog* Qwen-Image VAE-a.
MSE na VAE latentima. Jednostavno, stabilno, 1.3× jeftinije nego piksel-nivo
DiT rekonstrukcija, i ablation kaže da radi isto tako dobro ili bolje.

**Faza 2-2: kvantizacija.** Zameniš L∞ regularizaciju sa FSQ — Finite Scalar
Quantization. Svaka od 6 dimenzija se kvantizuje na jedan od [8, 8, 8, 5, 5, 5]
nivoa, što daje 64.000 mogućih kodova. Nema kodbuka koji se uči — FSQ je
bez optimizacije, samo zaokružiš na najbliži nivo.

Pre kvantizacije ubaciš 2D RoPE (rotary position encoding) da kodovi znaju
gde su prostorno. To je bitno pri proizvoljnim rezolucijama. Bez toga: 65.3.
Sa tim: 68.7. Learnable pozicionalni embedingi jedva pomažu (65.7) jer
o-težavaju optimizaciju kvantizacije.

Svaki vizuelni patch se takođe širi 2×2 (4 sub-patcha) preko attention-a
pre kvantizacije, pa se projektuje nazad. Ta 4 koda po patchu se procesuiraju
nezavisno — bitno jer nezavisnost ih čini boljim za downstream
reprezentaciono učenje (nema cross-patch zavezanosti u kodovima).

## Šta me stvarno iznenadilo

**Ne-učeni VQ pobeđuje.** FSQ (samo zaokruži na najbliži nivo, nema kodbuk
za treniranje) pobeđuje SimVQ (učivi kodbuk) za 2 poena. Testirali su i LFQ,
vanilla VQ, IBQ — isti pattern. U ovom settingu, kodbuk je teret jer uvodi
optimizacionu nestabilnost i codebook kolaps. Fiksna struktura FSQ-a izbegava
sve to. Prost princip ali ide protiv trenda sve složenijih kvantizacionih šema.

**VAE latent loss nad piksel lossom.** Izbor rekonstrukcijskog lossa je
fascinantan. Mislilo bi se da rekonstrukcija na nivou piksela (MSE + LPIPS na
sirovim pikselima) bi bila bolja za čuvanje detalja. Nije. Predviđanje latenta
pretreniranog VAE-a je jeftinije (1.3× naspram 2.3× za piksel MSE+LPIPS, 4×
za DiT) i zapravo radi bolje (68.7 vs 67.0 vs 65.8). Intuicija: VAE latenti
već kodiraju "bitne" vizuelne informacije, pa je regresija na njih bolji signal
nego sirovi pikseli koji uključuju šum, osvetljenje itd.

**Sweetspot za veličinu kodbuka.** 64.000 je dobro. 128.000 zapravo šteti
(68.3 vs 68.7). Zašto? Zato što sa fiksnim brojem trening slika, više kodova
= manja utilitzacija = otpušten kapacitet. Za ne-učeni FSQ ovo nije toliko
katastrofalno (68.3 vs 65.6 za učeni SimVQ pri sličnim veličinama), ali
sweetspot je stvaran.

**Koliko se tri lossa dopunjavaju.** Text loss sam: 61.3. Dodaj
self-distillation: 66.8. Dodaj rekonstrukciju: 68.7. Svaki daje značajan doprinos.
Self-distillation sprečava katastrofalno zaboravljanje originalnog vizuelnog znanja.
Rekonstrukcijski loss ubrizgava low-level detalj koji text supervizija sama ne
može da pruži. Bez bilo kog od njih, cela konstrukcija puca.

## Efikasnost

Evo praktičnog povrata. Tokom multimodalnog treninga, umesto da učitaš sirove
slike i pokreneš pun vizuelni enkoder svaki korak, precompute-uješ ViQ kodove
offline. Tokom treninga samo učitaš integer nizove i projektuješ ih u LLM
embedding prostor.

Za Qwen2.5-0.5B na 16k kontekstu: **78% ubrzanje forward prolaza**. Za veći
7B model na 4k: i dalje 46% brže. Preko svega, 20-70% ubrzanje treninga
zavisno od veličine modela i dužine konteksta.

Što je manji LLM, to je veći dobitak — jer je overhead enkodera veći deo
ukupnog compute-a za male modele.

A za skladištenje: 1920×1280 slika postaje 0.08 MB ViQ kodova (96×
kompresija). Isti odnos kao vrlo agresivan JPEG ali sa daleko boljim
rekonstrukcijskim kvalitetom.

## Gdje još uvijek zaostaje

ViQ prati 6B-parametarski InternViT-2.5 na prosečnim benchmarkovima, sa 1.3B
parametara i diskretnim kodovima. To je impresivno. Ali na OCRBench-u još uvijek
zaostaje (65.2 vs 69.2 za InternViT-6B). Papir je iskren o tome — to je
inherentno ograničenje diskretne tokenizacije, ne ViQ-specifičan flaw.
Agresivna kompresija u mali broj kodova će uvijek izgubiti neke
visokofrekventne detalje. Multi-scale ili rezidualna kvantizacija bi mogla
pomoći, i to označavaju kao budući rad.

Takođe, trening zahtijeva 128-256 A100 GPU-a. Nije nešto što repliciraš na
jednom workstationu.

## Verdikt

ViQ je jedan od onih papira gdje su ideje pojedinačno jednostavne (L∞
normalizacija, FSQ, VAE latent regresija, 2D RoPE) ali kombinacija i
stagewise trening recept čine da rade zajedno na način koji nijedna ne bi
sama. Proksimalna reprezentacija — to "meko slijetanje" pre kvantizacije —
je standout doprinos. Čisto, principijelno rješenje problema gubitka
informacija pri kvantizaciji koje nisam vidio negdje drugdje.

Rezultat je vizuelni enkoder koji izlazi kao integeri, radi na bilo kojoj
rezoluciji, čuva i semantiku i rekonstrukcijski kvalitet, i daje ti 20-70%
ubrzanje treninga. Praktičan alat, ne samo broj na benchmarku.

## Reference
- Papir: https://arxiv.org/abs/2606.27313
- Zvanični kod: https://github.com/yuxumin/ViQ
- Težine: https://huggingface.co/XuminYu/ViQ-weights
- Breakdown: `breakdown.md`
