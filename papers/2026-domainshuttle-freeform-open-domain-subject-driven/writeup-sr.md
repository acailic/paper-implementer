# Writeup — DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation

> Evo kako bih ovo objasnio prijatelju pre piva, ako me pita "šta si čitao?"

> **Jezici:** [English](writeup.md) · Srpski (ovaj fajl)

---

Jednostavna priča je otprilike ovako.

Recimo da imaš fotografiju svog psa i hoćeš da AI generiše video tog psa koji trči
kroz polje. To je in-domain — drži psa da izgleda potpuno isto, samo ga animiraj.
Postojeće metode za subject-driven video su prilično dobre u ovome.

Sada recimo da hoćeš video svog psa kao akvarel sliku. Ili kao 3D claymation
lik. Ili svog psa odštampanog na strani žutog školskog autobusa. To je
cross-domain — moraš zadržati psa prepoznatljivim (uši, boja, oblik) dok
potpuno menjaš stil, osvetljenje i medij oko njega. Skoro niko ne radi na ovome,
a metode koje postoje ili samo copy-paste-u referentnu fotku u video ili gube
subjekat potpuno kad pokušaju da ga transformišu.

DomainShuttle kaže: hajde da radimo oba, i to istim modelom.

## Arhitektura običnim jezikom

Cela fora je u **rastavljanju**. U standardnom video diffusion transformeru,
tokeni referentnih slika i video tokeni su pomešani u istim attention slojevima,
procesuirani istim projekcijama, pozicionirani u istom RoPE (Rotary Positional
Encoding) prostoru. DomainShuttle to razdvaja u tri odvojena poteza.

**Potez 1: Domain-MoT (nezavisne attention grane).** Daj video tokenima i
referentnim tokenima njihove sopstvene Q/K/V projekcione matrice umesto da dele
iste. Video grana nastavlja da radi ono što bazni model već ume — generiše dobar
video. Referentna grana se specijalizuje za ekstrakcija osobina subjekta.
Povrh toga, referentna grana dobija dodatni conditioning signal: *domain
atribut* (real-world čovek, real-world objekat, fantasy lik, pozadina) koji joj
kaže kakav je tip subjekta. Video grana vidi samo vreme; referentna grana vidi
vreme *plus domain*. To znači na inferenci možeš zameniti domain atribut da
promeniš stil bez diranja bilo čega drugog.

**Potez 2: VR-DualRoPE (odvojeni pozicioni prostori).** Normalno se referentne
slike tretiraju kao dodatni video frejmovi u RoPE prostoru — dobijaju temporalni
indeks kao svaki drugi frejm. To je pogrešno iz dva razloga: različiti referentni
subjekti nemaju temporalni odnos međusobno, i više fotografija *istog* subjekta
bi trebalo da budu povezane, a ne razbacane kroz vreme. VR-DualRoPE stavlja sve
referentne tokene u potpuno odvojeni RoPE prostor sa temporalnim indeksom
fiksiranim na nulu, i koristi prostorne offsete da razdvoji različite subjekte
dok drži slike istog subjekta blizu jedna drugoj.

**Potez 3: CCL (cross-pair konzistencija).** Za svaki trening video imaju
više setova referentnih slika (različiti uglovi, osvetljenje, cropovi). Tokom
treninga provuku dva različita referentna seta kroz model na *istom nivou šuma*
i primoraju predikcije da se poklapaju. Jedna grana je zamrznuta, jedna se
trenira. Ovo uči model: "ove dve različite fotke prikazuju isti subjekat — nauči
šta je zajedničko, ignoriši što je različito." Rezultat je da model uči intrinsične
osobine subjekta (identitet, oblik, boja) umesto da overfittuje na specifične
artifakte jedne referentne slike.

## Brojevi koji bitno

Glavni rezultat: **CD-Score od 0.861**, što je 18.7% bolje od Kling 1.6 (0.725)
i 54.5% bolje od narednog najboljeg open metoda (FFGO na 0.558). CD-Score meri
koliko dobro intrinsične osobine subjekta prežive domain transformaciju — baš
onaj metrič koji ovaj papir cilja.

In-domain metrike (DINO-I, CLIP-I) su kompetitivne ali ne uvek apsolutno najbolje.
To je u redu — namerna trgovina. Model odustaje možda 1-2% in-domain fidelnosti
za masivan gain u cross-domain sposobnosti. Za svakog ko hoće da radi kreativni
rad sa video generacijom, to je odlična ponuda.

Ablacija priča čistu priču:

```
CD-Score:  0.697 → 0.715 → 0.783 → 0.813 → 0.861
           naive   +dual   +MoT    +RoPE   +CCL
```

Domain-MoT je najveći pojedinačni doprinos. Dodavanje domain-aware AdaLN-a na
referentnu granu je ono što uopšte otključa cross-domain transformaciju — bez
njega, naive metoda jednostavno ne uspeva da prebaci subjekte u target domain.
Ostala dva modula dodaju dalje poboljšanja.

## Dve stvari koje su mi bile zanimljive

**CCL je o kontrollability-ju, ne o fidelnosti.** Ovo me iznenadilo. Očekivao
sam da CCL poboljša i cross-domain i in-domain metrike podjednako. Umesto toga,
on baci CD-Score za 5.9 procentnih poena ali jedva dodiruje CLIP-I (+0.3%) i
DINO-I (+1.5%). To znači da CCL specifično uči model *šta da transformiše a šta
da sačuva* — ne samo da model bolje prepoznaje subjekte, nego da bolje razdvaja
intrinsične osobine od domain-specifičnih. To je suptilnija sposobnost nego što
sam očekivao od onoga što je u suštini konzistencija regularizaciona gubitka.

**VR-DualRoPE blago pogoršava CLIP-I.** Zvuči kontraintuitivno — bolji prostorni
model bi trebalo da pomogne sličnost, zar ne? Ali subject-decoupled offset strategija
približava slike istog subjekta u RoPE prostoru, što znači da model tretira klaster
umesto individualnih visoko-fidelnih kopija. CLIP-I meri frejm-level sličnost sa
jednom referentnom slikom, pa klasteriranje može zapravo smanjiti score.
Subject-level metrike (CD-Score, DINO-I) se poboljšavaju, ipak. Podsetnik da
optimizacija za pravi metrič bitno utiče — frejm-level sličnost nije isto što i
subject-level konzistencija.

## Trening recept

Dva etapa. Prvo, 2,000 koraka na 200K image personalizacionih podataka da bi bazni
model dobio osnovnu subject-svesnost. Zatim 12,000 koraka na 750K video
personalizacionih podataka za pravi trening. Cross-attention je zamrznut kroz celu
drugu etapu da se sačuva text-following sposobnost. Ukupno: 30,000 GPU-sati na
14B modelu. Skupo, ali zvanični kod je Apache 2.0 i uključuje trening skripte,
što reprodukciju čini daleko izvodljivijom nego kod većine video generacionih
papira.

Data pipeline je pravi radni konj. Gradnja cross-pair referentnih setova zahteva
Grounding-DINO za detekciju, SAM2 za segmentaciju i MLLM za kvalitetno
filtriranje. Ditto-1M dataset takođe pruža editing parove (referenca → editovan
video) kao augmentaciju. Bez Ditto-1M, model i dalje radi (CD-Score 0.823 vs
0.861) i dalje pobeđuje sve baselines — pa je osnovna metoda robustna, editing
data je samo bonus.

## Šta je bilo teže za razumeti nego što sam mislio

Domain atribut anotacija je suptilna. Papir kaže da domain atributi odgovaraju na
"atribute subjekta u *generisanom* videu, ne u referentnim slikama." Tako da ako
imaš fotku prave osobe i hoćeš da generišeš nju kao fantasy lik, domain atribut
je "fantasy subjekat" — ne "real-world čovek." Atribut opisuje kuda ideš, ne odakle
polaziš. Trebalo mi je ponovno čitanje da to uhvatim.

Takođe, CCL mehanizam koristi *zamrznutu* granu, ne momentum encoder ili EMA.
Zamrznuta grana `G*_θ` se nikad ne ažurira — to je snapshot. Samo trening grana
`G_θ` uči. Ovo je prostije od tipičnih consistency regularization postavki
(BYOL, SimSiam) i izbegava representational collapse problem po konstrukciji —
zamrznuti target pruža stabilan learning signal. Čist dizajn.

## Reference
- Papir: https://arxiv.org/abs/2606.26058
- Zvanični kod: https://github.com/HKUST-C4G/DomainShuttle
- Projekat: https://cn-makers.github.io/DomainShuttle/
- Breakdown: `breakdown.md`
