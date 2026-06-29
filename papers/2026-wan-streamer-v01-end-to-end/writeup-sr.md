# Wan-Streamer v0.1 — Objasnjeno prijatelju

> **English version:** [writeup.md](writeup.md)

---

## U jednoj rečenici

Wan-Streamer je jedan Transformer od Alibabe koji radi interakciju licem u lice u realnom vremenu — sluša tvoj govor, gleda tvoj video, i odgovara sa sinhronizovanim govorom i videom svoje animirane avatare, a sve to bez da spaja zasebne module za prepoznavanje govora, jezički model, sintezu govora i renderovanje avatara. Svaki komponenta je dizajniran da bude striktno kauzalan (procesira slijeva nadesno), tako da model može da strimuje 160ms komade na 25 FPS-a uz pun konverzacijski kontekst. Na inferenci se dijeli na dva GPU-a ("thinker" koji radi percepciju i dekodiranje, i "performer" koji generiše skupi video/audio), i postiže ~200ms latenciju modela i ~550ms ukupno uključujući mrežni put. Trenutni v0.1 radi na 192p rezoluciji — namjerno dokaz koncepta — a oni tvrde da skaliranje na više rezolucije je jednostavno.

---

## Problem

Zamisli da si na video pozivu sa AI-jem. Hoćeš da to izgleda kao razgovor sa pravom osobom: pričaš, on sluša i kimne glavom, prekidaš ga, on stane, primijeti da si uzeo kafu i komentariše to. Sad razmisli kako bi to izgradio.

Očevidan pristup — i ono što većina sistema zapravo radi — je da povežeš hrpu specijalizovanih modula: prepoznavanje govora te sluša, jezički model odlučuje šta da kaže, sintetizator govora generiše audio, i avatar renderer animiše lice. To radi ok za razmjenu poruka, ali puca za prirodnu konverzaciju:

- **Latencija se gomila.** Svaki modul treba vrijeme, a rade sekvencijalno. Korisnik čeka.
- **Nema sinhronizacije.** Pokreti usana avatara se generišu nakon govora, pa su uvijek malo izvan takta. Možeš to zakrpati post-procesiranjem, ali nikad ne djeluje baš pravo.
- **Nema ponašanja slušanja.** Dok ti pričaš, sistem samo snima — avatar ili se smrzne ili ponavlja generičku animaciju. Prava osoba bi kimala, održavala kontakt očima, reagirala.
- **Prekidanje je čudno.** Da bi obradio "čekaj, da ti završim misao", sistem mora da rasturi TTS pipeline usred riječi i krene ispočetka.

Osnovni problem: ovi kaskadni sistemi nikad nisu bili dizajnirani za strimovanje. Dizajnirani su za turn-based interakciju gdje ti završiš pričanje, onda AI razmišlja, onda AI priča. Ljudi ne tako rade.

---

## Ideja

Odgovor Wan-Streamera: baci sve module. Izgradi jedan jedini Transformer koji radi sve — tekst, audio i video i na ulaznoj i na izlaznoj strani. Dizajniraj svaki komponentu da bude kauzalan (može da gleda samo u prošlost, nikad u budućnost). Onda cela stvar može da radi kao jedan kontinuirani tok, baš kao ljudska konverzacija.

Osnovni princip dizajna koji oni zovu "streaming contract": svaki komponenta mora da radi kauzalno, svaki novozapaženi komad mora biti upotrijebiv odmah, i svaki generisani komad mora biti emitovan i dodat u istoriju interakcije. Nema čekanja na kompletnu izjavu, nema batchovanja, nema naknadnog usklađivanja.

---

## Kako radi (intuitivno)

Zamisli model kao da procesira jedan dugi niz koji izmjenjuje sve:

```
[tekst korisnika] [audio frejmovi korisnika] [video frejmovi korisnika] [tekst agenta] [audio latent agenta] [video latent agenta] [tekst korisnika] ...
```

Na svakih 160ms (4 frejma na 25 FPS-a):

1. **Vidi korisnika.** Audio i video VAE kompresuju najnoviji komad korisnika u latent tokene. Oni su striktno kauzalni — gledaju samo prošle frejmove, ne buduće.

2. **Razmisli.** Transformer radi kauzalni prolaz preko novih tokena plus pune istorije konverzacije (sačuvane kao KV keš). Proizvodi tekst (rijec po rijec, kao jezički model) i predviđa audio+video latente (koristeći flow matching, poput difuzionog modela ali brže).

3. **Govori i pokaži.** Audio i video dekoderi pretvaraju latente u stvarni zvuk i piksele. Odlaze korisniku odmah.

4. **Zapamti.** Generisani latenti se dodaju u istoriju konverzacije, tako da slijedeći korak ima pun kontekst.

Tricky dio je flow matching za audio i video. Umjesto sporog difuzionog procesa, koriste conditional flow matching — kreću od šuma i iterativno ga preoblikuju prema cilju. Audio i video se "denoiziraju" zajedno (kondicionirani na isti kontekst), što znači da su pokreti usana i govor prirodno sinhronizovani jer su generisani zajedno, a ne sašiveni naknadno.

Za inferenciju, dijele model na dva GPU-a:
- **Thinker GPU:** kodira ulaz korisnika, radi Transformer za tekst/stanje, dekodira izlaz prethodnog koraka
- **Performer GPU:** radi flow-matching solver za slijedeći korak audio+video

Komuniciraju preko KV-keš razmjene. Ključna ideja: na koraku k, thinker dekodira izlaz koraka k-1 dok performer generiše latente koraka k. Ovo pipelining sakriva većinu latencije.

---

## Šta me iznenadilo

1. **Nema nijedne ablacije.** Za papir koji tvrdi jake arhitektonske stvari (kauzalni VAE-i, blok-kauzalna pažnja, zajednički flow matching, thinker-performer split), nemati nijednu ablaciju je zapanjujuće. To je v0.1 tehnički izvještaj, ali čak i osnovna poređenja bi pomogla.

2. **Latencija je zaista konkurentna.** ~200ms na strani modela za pun audio+video izlaz je impresivno, čak i na 192p. Sistema samo za govor kao što je Moshi prijavljuju slične brojeve a nemoraju da generišu video.

3. **Ponašanje "slušanja" je naučeno, ne programirano.** Model nije eksplicitno treniran sa labelama "kimni kad korisnik priča". Naučio je to iz interleaveovanih interakcijskih podataka gdje ljudski sagovornici prirodno pokazuju takva ponašanja. Prilično kul.

4. **Papir je neobično iskren o granicama mjerenja.** Oni eksplicitno ukazuju da je poređenje njihove full end-to-end latencije sa parcialnim metrikama drugih sistema (samo model, prvi paket, samo renderer) obmanjujuće, i tabeliraju šta svaki sistem zapravo mjeri. To je dobra naučna praksa.

5. **Rolling distilacija za smanjenje jazova trening/test.** Tokom distilacije, student model se trenira na vlastitoj generisanoj istoriji umjesto na izlazima nastavnika, koristeći distribution matching za poravnanje trajektorija. Lijepa tehnika preuzeta iz self-forcing literature.

6. **192p i oni su ponosni na to.** Većina papira bi pokušala da sakrije 192p izlaz. Oni su to stavili naprijed kao dizajnerski izbor — dokaz koncepta za arhitekturu, ne tvrdnja o kvalitetu. Osvježavajuće.

---

## Reference

- Papir: https://arxiv.org/abs/2606.25041
- Projekat: https://wan-streamer.com/
- Srodno: Wan2.1 video generacija (osnovna arhitektonska familija)
- Srodno: Self-forcing i distribution matching (tehnike distilacije korištene u fazi 3)
- Srodno: Moshi (full-duplex govor, bez videa), VASA-1 (avatar vođen audio-em, bez dijaloga), TalkingMachines (video vođen audio-em sa eksternim LLM-om)
