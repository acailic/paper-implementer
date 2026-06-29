# Writeup — PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation

> Evo kako bih ovo objasnio prijatelju pre piva, ako me pita "šta si čitao?"

> **Jezici:** [English](writeup.md) · Srpski (ovaj fajl)

---

Jednostavna priča je otprilike ovako.

Video si te generativni modeli — Sora, Wan, Cosmos — pumpaju nevjerovatno lijep kadrove. Zamisli sad da si robota istraživač i želiš da koristiš jedan od ovih kao svjetski simulator. Ubačiš sliku robotske ruke i prompt "pokupi crvenu šolju i stavi je na policu." Ono što dobiješ izgleda sjajno — dok ne primjetiš da gripper fazonira kroz šolju, ili da šolja lebdi u zraku jedan kadar, ili da robot gura šolju ali šolja se niti ne pomakne.

Lijepi kadrovi ≠ fizikalna plauzibilnost. A za robotiku, to je dealbreaker.

Ovaj rad ima opservaciju koja je skoro frustrirajuće očigledna retroaktivno: fizikalni grešci u robotskim manipulacijskim videima dolaze u dva oblika, i žive na specifičnim mjestima. **Lokalno**, pojedinačne tačke skaču (diskontinuirane trajektorije). **Globalno**, relacije između objekata pucaju (gurnuti objekt stoji, uhvaćeni objekt se klizi). I sve se to dešava oko **kontaktnih zona** — vrhova gripa, površina objekata, pokretnih dijelova. Pozadinski pikseli ne trebaju fizikalnu superviziju.

Dakle, rješenje: nađi gdje je akcija, pa primijeni dva lossa — jedan za lokalno kretanje, jedan za globalne relacije — i to samo na tim regijama. To je PhysisForcing. Dva lossa tokom treninga, nula dodatnog troška na inference-u.

## Kako nalaze gdje fizika važi

Prvo pokrenu point tracker (CoTracker3) na videu da dobiju guste trajektorije — gdje se svaki piksel kreće kroz vrijeme. Zatim koriste depth estimator (Depth-Anything-V2) na prvom kadru da utvrde šta je u foregroundu. Kombinuju magnitudu kretanja sa foreground blizinom, adaptivno pragom dobiju spatiotemporalnu masku koja ističe manipulatore, objekte i kontakt zone. Pozadina je van.

## Dva lossa

**Pixel-level trajektorijski loss.** Uzme jedan intermedijarni sloj DiT-a (ne prvi, ne zadnji — onaj u sredini, empirijski najbolji). Projuri kroz mali MLP. Koristi feature prvog kadra kao query, ostale kao key. Računa similarity mape, softmax, i izvlači predviđene lokacije tačaka kao težinske prosjeke prostornih koordinata. Uporedi te predikcije sa ground-truth trajektorijama od tracker-a, maskirano na fizikalne regije.

Šta ovo radi: forsira DiT-ove interne feature da kodiraju glatko, kontinuirano kretanje na kontakt tačkama. Nema više gripa koji se teleportiraju ili objekata koji se pojavljuju.

**Semantic-level relational loss.** Pokrene frozen video understanding enkoder (V-JEPA 2) na istom klipu. On proizvodi token reprezentacije koje prirodno hvataju relacije objekata — enkoder zna da gripper i uhvaćena šolja trebaju biti čvrsto spojeni jer je treniran self-supervised na videima. Zatim uzme DiT feature sa istog srednjeg sloja, projuri u enkoderov prostor, i uporedi pairwise cosine similarity matrice. Isti mask, isti odabrani tokeni. Forsira DiT da replicira enkoderovu relacionu strukturu.

Šta ovo radi: osigurava da regije koje se trebaju kretati zajedno zaista i rade, na semantičkom nivou. Čak i ako su pojedinačne piksel trajektorije u redu, ovo hvata slučajeve gdje je ukupna interakcija pogrešna — poput gurnutog objekta koji stoji dok robot je očito napravio kontakt.

## Dve stvari koje su me stvarno iznenadile

Prvo — **srednji slojevi su kraljevi.** Oni swipaju koji DiT blok da aligniraju (sloj 10, 15, 25 od ~40). Sloj 15 pobjeđuje odlučno (85.2 naspram 83.9 naspram 83.2 na PAI-Bench). Razlog je čist: rani slojevi nose plitke appearance feature, kasni slojevi su već specijalizovani za noise predikciju i teško se skreću. Srednji slojevi imaju i semantičku strukturu i plastičnost koja treba. Ovo djeluje kao generalni princip za bilo koji pristup aligniranja na intermedijarnim feature-ima, ne samo ovaj rad.

Drugo — **razblaživanje pozadine je stvarno, ne samo teorijski očekivano.** Ablacija radi ovako: primijeni iste ta dva lossa uniformno na sve tokena (bez maske) naspram samo na fizikalno informativne tokena. Uniformno pomaže (44.8 → 46.0), ali maskirano je bolje (44.8 → 47.5). Razlika od 1.5 poena dolazi u potpunosti iz task-oriented metrika (35.4 → 38.9). Pozadinski pikseli aktivno štete fizikalnom učenju jer razblažuju gradijent signal. Ne bih očekivao da je to toliko izraženo — maska nije samo trik za efikasnost, ona je kvalitetni knob.

## Brojevi koji važe

Na R-Bench-u (650 prompt-ova robot manipulacija + lokomocije), PF-Cosmos (Cosmos3-Nano treniran sa PhysisForcing-om) dobija 63.8, pobjeđuje sve uključujući komercijalni Wan2.6 (60.7). PF-Wan stiže do 62.0 na 14B backbone-u. Naspram vanilla fine-tuning-a, dobici su +4.1 i +7.1 — a naspram sirovih base modela, +5.4 i +22.3.

Na PAI-Bench robot domenu (174 real-world prompt-ova, suđeno od Qwen3-VL-235B), PF-Cosmos score-uje 85.2 ukupno, opet pobjeđuje komercijalni Wan2.5 (81.0) i najbolji robota-specific model Abot-PhysWorld (84.9).

Zero-shot EZS-Bench rezultat je onaj koji nalazim najuvjerljivijim. Trening-independent — 196 neviđenih kombinacija robot-zadatak-scena, nema preklapanja sa trening podacima. PhysisForcing i dalje poboljšava oba backbone-a (79.0→80.5, 80.3→81.1), što sugerira da su fizikalni prijevi zapravo generaliziraju umjesto da memoriraju trening distribucije.

## Iza generacije: ovo pomaže robotima

Evo gdje rad ide dalje od "napravili smo ljepše videe." One ubace PhysisForcing-trenirani svjetski model u dvije robotičke pipeline-e:

**WorldArena action planner:** Svjetski model predviđa buduće kadre, inverzni dinamika model dekodira te u akcije, a robot izvršava u simulaciji. PhysisForcing podiže closed-loop success sa 16.0% na 24.0%, pobjeđuje najbolji specijalizovani world-model planner WoW (20.5%).

**Fast-WAM downstream politika:** Koriste PhysisForcing-trenirani Wan2.2-TI2V-5B kao video backbone unutar world-action modela. Prosječna uspješnost politike raste sa 68.2% na 72.8% na RoboTwin 2.0 zadacima. Najveći dobici su na najkontakt-bogatijim zadacima: stavljanje prazne šolje (+21.5%) i pritisak stajlera (+11.0%). To su tačno scenariji gdje fizikalna plauzibilnost najviše znači.

Argument je jasan: fizikalno alignirani video modeli uče bolje interne reprezentacije za robotiku, ne samo bolje izgledajuće outpute.

## Šta bih pitao

Trening podaci su 500K klipova filtrirano iz RoVid-X-ovih 4M. To je dosta filtriranja. Koliko poboljšanja dolazi od čistih in-domain podataka naspram fizikalnih loss-ova specifično? Oni djelimično adresiraju ovo sa vanilla fine-tuning baseline-om, ali eksperiment kontrolisan podacima (isti podaci, različiti loss kombinacije) bio bi čišći.

Takođe, eksperimenti generacije su svi image-to-video (text + image conditioning), ne action-conditioned. WorldArena i Fast-WAM eksperimenti koriste action-conditioned modele, ali su to downstream evaluacije sa drugačijim backbone-om (Wan2.2-TI2V-5B). Volio bih vidjeti PhysisForcing primijenjen direktno na action-conditioned generativni model i evaluiran na kvalitetu generacije.

Konačno, V-JEPA 2 kao semantički teacher je jak ali donekle proizvoljan izbor. Da li je relaciona struktura ono što važi, ili je V-JEPA 2 specifično dobar? Teacher-ablacija (zamijeni sa DINOv2, InternVideo, itd.) bi razjasnila ovo.

## Reference
- Rad: https://arxiv.org/abs/2606.28128
- Project page: https://dagroup-pku.github.io/PhysisForcing.github.io/
- Breakdown: `breakdown.md`
