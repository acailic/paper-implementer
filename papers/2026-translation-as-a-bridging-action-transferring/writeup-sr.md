# Writeup — Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots

> Evo kako bih ovo objasnio prijatelju pre piva, ako me pita "šta si čitao?"

> **Jezici:** [English](writeup.md) · Srpski (ovaj fajl)

---

Evo postavljanja. Imaš robota sa dve ruke i hvataljke. Hoćeš da ga naučiš da otvara mikrotalasne rerne, briše ploče, kači šolje na kačke, vadi punjače iz utičnica — stvari koje ljudi rade svakog dana a da ne razmišljaju o tome. Očigledno rešenje: snimi ljude kako to rade, izvadi pokrete njihovih ruku, i treniraj robota na tome.

Osim što ne radi dobro. I ovaj papir objašnjava zašto, pa onda pokazuje ispravku koja je u suštini trivialna s retrospektivom.

## Problem sa ljudskom rotacijom zgloba

Mainstream pristup — koriste ga EgoMimic, GR-3, EMMA, i uglavnom svi u ovom prostoru — tretira ljudsku ruku kao još jednu robotsku ruku. Pokreneš hand pose estimator na egocentričnom videu, dobiš poziciju i rotaciju zgloba (6 stepeni slobode), i to nahraniš robotskoj politici kao da je čovek bio 7-DoF ruka sa hvataljkom.

Dva problema. Prvo, procene rotacije zgloba iz pose estimatora su bučne. Roll i posebno pitch — estimator jednostavno nije dovoljno siguran. Drugo, i ovo je ono što zaista ima težinu: ljudski prsti i paralelne hvataljke se ne kreću na isti način. Kad držiš ručku vrata prstima, tvoj zglob može da se rotuje na sve strane dok prsti održavaju kontakt. Paralelna hvataljka to ne može — rotacija hvataljke direktno menja kontakt. Rotacija zgloba koja je potpuno normalna za ljudsku ruku prerušava se u izobličenu, beskorisnu pozu na robotskoj hvataljci.

Papir to pokazuje kvalitativno i to je ružno. Robot sa 6DoF ljudskim akcijama se uvija u čvorove. Papir to pokazuje i kvantitativno — 38% progresa naspram 49% za njihov pristup.

## Ispravka: samo baci rotaciju

Njihova ključna ideja je sramotno prosta. Zaboravi rotaciju u potpunosti. Zadrži samo translaciju zgloba — smer i rastojanje koje se ruka pomera — u frame-u kamere na glavi. I čovek i robot vide svet iz otprilike iste kamere na glavi, pa je relativna translacija zgloba zajednički jezik. Fizički je smislena, robustna je prema buci iz pose estimatora (translacije su mnogo pouzdanije od rotacija), i radi na isti način bez obzira da li imaš prste ili paralelnu hvataljku.

Zovu ovo "bridging action": `a3D-wrist`. Za bimanual setup to je 6 brojeva po timestep-u (3 po ruci) — samo koliko daleko i u kom smeru se svaki zglob pomera.

## Interleaved action tokeni

Ovo je arhitektonski doprinos i zanimljiviji je nego što zvuči na prvo čitanje.

Model je π0-stil VLA — vision-language-action model sa pre-treniranom VLM kičmom i posebnim action transformer-om koji generiše akcije preko flow matching-a. Trik je u tome kako rukuju time da različiti izvori podataka imaju različite dostupne akcije:

- In-the-wild ljudski video: imaš samo translaciju zgloba (nema hvataljke, nema end-effector-a)
- In-lab ljudski podaci: imaš translaciju + anotiranu hvataljku (ruka otvorena/zatvorena)
- Robot tele-operacija: imaš sve — translaciju, 6DoF end-effector, i hvataljku

Slažu action token-e u specifičnom redosledu: `[bridging → 6DoF end-effector → hvataljka]`. Bridging signal dolazi prvi. Ovo nije slučajno — znači da 6DoF end-effector token-i mogu da prisustvuju bridging token-ima, pa znanje o *gde da se pomeraš* (naučeno iz ljudskih podataka) teče direktno u *kako da pomeriš robotsku ruku* (potrebno za izvršenje). Komponente koje nedostaju su maskirane u attention-i i isključene iz loss-a.

## Trening pipeline

Tri stage-a, i svaki je bitan:

**Stage I: Pre-train samo na ljudima.** ~600 sati ljudskih manipulacija — mešavina EgoDex isječaka, out-sourced slobodnih domaćih zadataka, i in-lab snimanja. Nadgleda se samo bridging akcija. Model nikad ne vidi robotsku akciju. Ovo je čisto razumevanje ljudskog kretanja.

**Stage II: Co-train ljudi i roboti.** Dodaju ~72 sata robot pick-and-place podataka (generičko, preko 100 objekata) plus ~3 sata po zadatku specifičnih ljudskih demonstracija (neko otvara mikrotalasnu, briše, itd.). Na robot podacima, nasumično zamenjuju između a3D-wrist i a6D-eef kao prediction target. Ovo je kritično — forsira model da ground-uje bridging reprezentaciju u izvršive robotske akcije. Bez ovoga, bridging signal samo lebdi u latentnom prostoru i nikad se ne povezuje sa stvarnom robotskom kontrolom.

**Stage III: Few-shot robot post-training.** Samo 10 robotskih demonstracija po zadatku. Evo gde vidiš plod pre-treninga.

## Rezultati koji bitno utiču

Glavni broj: trening samo na robot pick-and-place podacima te donekle 0% uspeha na svih 15 evaluacionih zadataka. Dodaj ljudske bridging akcije preko co-training-a: do 31% uspeha. Dodaj ljudski pre-trening na vrhu: 38% uspeha. Dodaj 10 robot dema po zadatku: 55% uspeha. Robot zaista uči manipulaciju koju nikad nije video u robot podacima.

Najuočljivija komparacija je bridging naspram 6DoF ljudskih akcija (Tabela 2). Isto setup, isti podaci, isti model — samo drugačija akciona reprezentacija za ljudske podatke. Bridging pobeđuje sa 11 procentnih poena na progresu i 13 na uspehu. A kvalitativna razlika je dramatična: jedan proizvodi stabilnu, prirodnu manipulaciju; drugi izobličene, izuvijane poze zgloba.

Pa onda ima ablacije koja zaista bitno utiče (Tabela 4). Uklanjanje nasumične bridging supstitucije na robot podacima za vreme co-training-a srusi uspeh sa 38% na 12.5%. Model apsolutno mora da bude eksplicitno primoran da poveže zajedničku bridging reprezentaciju sa izvršivim robotskim akcijama. Neće to uraditi sam od sebe.

I eksperiment gornje granice (Tabela 5) je u tišini jedan od najzanimljivijih rezultata. Uzmu prave robotske demonstracije, skinu rotaciju i prednosti opservacije, i treniraju sa istim objektivom kao za ljudske podatke. Performanse skaču na 73.5% progresa i 55.8% uspeha — znatno iznad defaultnih 59.8%/38.3%. Bridging reprezentacija ima pravog prostora za rast. Usko grlo nije reprezentacija, već jaz između ljudskog i robotskog embodiment-a.

## Šta sam našao najzanimljivijim

Rezultat alignment-a loss-a (Figure 9) je tihi dragulj. Pre-trening samo na bridging signalu — ne-izvršivoj, translacijskoj reprezentaciji — daje *niži* trening loss za i 6DoF end-effector akciju i hvataljku za vreme co-training-a. Model koji je naučio "kuda se ruke kreću" u ljudskim videima brže konvergira na "kako da pomera robotske ruke". Objektivni pejzaži su poravnati, iako je pre-trening objektiv potpuno ne-izvršiv. Ovo je jak dokaz da bridging reprezentacija hvata nešto fundamentalno o manipulaciji koje transcendiraa embodiment.

Failure case-ovi su iskreni i informativni. Zadaci koji zahtevaju preciznu rotaciju end-effector-a pri kontaktu — ubacivanje slamke u šolju, otvaranje fioka — su gde pristup puca. Robot pokazuje jasnu namenu zadatka, stigne do pravog područja, ali ne može da izvrši kritični rotacioni korak. Ovo je tačno onaj trade-off koji očekuješ kad namerno baciš rotaciju. Autori su iskreni oko toga i kažu da dodavanje ograničenih, pouzdanih rotacionih signala dolazi kao budući rad.

## Šta bih ja pravio od ovoga

Najkorisnija ideja za odmah je nasumična supstitucija prediction target-a za vreme co-training-a. Ideja da možeš da forsiraš model da ground-uje zajedničku reprezentaciju u specifičan akcioni prostor nasumičnim zamenjivanjem target-je je generalizabilna daleko ovog papira. Ako imaš bilo koji multi-embodiment setup — različiti roboti, različite hvataljke, različiti akcioni prostori — možeš da primeniš isti trik: treniraj na zajedničkoj reprezentaciji, ali nasumično zamenjuj embodiment-specifičnu akciju kao target da forsiraš vezivanje.

Bridging reprezentacija sama po sebi je agnostična po platformi. Trebalo bi da radi na jedno-ručnim setup-ima, mobilnim manipulatorima, čak humanoidima — bilo šta sa kamerom na glavi i zglobovima. Interleaved token dizajn sa attention masking-om je čisto rešenje za heterogeni akcioni problem koje drugi papiri rešavaju ružnijim padding-om i konkatencijom.

## Reference
- Papir: https://arxiv.org/abs/2606.28133
- Projekat: https://translation-as-a-bridging-action.github.io/
- Breakdown: `breakdown.md`
