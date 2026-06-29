# Writeup — Are We Ready For An Agent-Native Memory System?

> Evo kako bih ovo objasnio prijatelju pre piva, ako me pita "šta si čitao?"

> **Jezici:** [English](writeup.md) · Srpski (ovaj fajl)

---

Jednostavna priča je otprilike ovaka.

Agent memorija je izrasla iz "pretraži transkript pa vidi šta iskoči" u pravi mali bazični sistem — negde se stvari zapakuju, negde se traže, negde se menjaju, negde se brišu. To više nije samo neki RAG iznad istorije četa. To je infrastruktura.

Problem je što je sva evaluacija ostala u prošlom veku. Ljudi mere krajnji F1 i kažu "Mem0 je 21, Zep je 84" — i stvar tu. Niko ti ne zna reći *zašto*. Je li to zato kako smeštaju memoriju? Kako je vade? Kako je održavaju? Ćorak posao.

Ovaj rad kaže: stani. Rastavi svaki sistem memorije na četiri dela, pa meri svaki posebno.

## Četiri modula

Svaki sistem memorije agenta na svetu je zapravo jedan te isti obrazac, samo različito popunjen:

- **R** — kako pamtiš i gde to stojiš (flat tekst? graf? kompozit?) i fizički gde (u kontekstu? u vektor bazi? u više engine-a odjednom?)
- **S** — kako sirov dijalog postaje memorija (samo nalepiš redove? izvučeš činjenice? parsiraš u triplets?)
- **Q** — kako nađeš što treba kad dođe upit (pažnja? vektorska pretraga? šetnja po grafu? LLM kao planer? hibrid?)
- **U** — kako se to održava kroz vreme (append sa timestampovima? FIFO izbacivanje? LLM spaja slično? offline fine-tuning?)

Autori ovo zapišu kao `M_sys = ⟨R, S, Q, U⟩` i onda — i tu je zapravo doprinos — krenu da benchmark-iraju svaki slot nezavisno, sa pravim evidence-level merama i sa cenom. Ne samo "da li je odgovor dobar", nego "da li si uopšte izvukao dokaz koji treba", i "koliko te to koštalo u sekundama po upitu".

I testiraju 12 ozbiljnih sistema na 5 različitih workload-a. Odjednom vidiš gde koji puca.

## Dve stvari koje su me stvarno iznenadile

Prvo — i ovo je meni bio najveći "aha" trenutak — **pretraga memorije nije problem rangiranja, nego problem sastavljanja dokaza.**

Eksperiment RQ2 je tu ključan. SimpleMem dobije Recall@1 (prvi pogodak) od 39 — fenomenalno u tom jednom pogledu, ume odmah da izvuče onaj jedan očigledno bitan red. Ali čim povećaš budžet na Recall@5 i Recall@10, ili čim se dokaz nalazi dalje u prošlosti, A-MEM i MemTree ga potpuno zgaze (69.5/85.9 i 59.7/80.5 naspram flat Embedding RAG koji pada sa 37.1 na 7.4 F1).

Šta to znači? Teški slučaj nije naći *prvu* relevantnu stvar. Teško je skupiti *kompletan* rasuti skup memorija potrebnih da bi se odgovorilo — i tu se isplati eksplicitna struktura, linkovi, hijerarhija. Flat similarity je oružje kratkog dometa. To je nešto što nosiš sa sobom i kad zatvoriš papir.

Drugo — **cenu ne gura struktura, gura je obim održavanja.** Ovo je O7 i zapravo preokreće celu debatu "graf vs vektor". Ljudi misle da je strukturirana memorija (grafovi, hibridi) skupa zato što je strukturirana. Podaci kažu suprotno: skupa je kad svaki upis **propagira globalno**. LightMem i MemTree ostaju jeftini jer su im upisi lokalni — nova činjenica dirne samo svoje podstablo. Cognee i Zep postaju skupi (116s, 155s) jer svaki upis okine konsolidaciju celog grafa. Struktura je u redu. **Globalna prekalkulacija je neprijatelj.** To je konkretan inženjerski princip, ne samo broj na tabeli.

## Šta sam naučio kada sam to sam iskodirao

Kad kreneš da praviš ta četiri modula ispočetka (vidi `implementation/`), isplivaju tri stvari koje papir implikuje ali ne kaže glasno:

**Moduli su jače spregnuti nego što taksonomija izgleda.** Ne možeš baš slobodno da biraš R, S, Q, U nezavisno. Na primer, timestamp multi-versioning (to je U izbor) radi samo ako ti R izlaže verzije po entitetu — inače ne možeš da vežeš ispravljenu činjenicu za entitet koji ažuriraš. Ili, balansirana hibridna pretraga (Q) se isplati samo ako ti je S sačuvao dovoljno sirovog teksta po kom BM25 može da traži. Tuple je *sočivo*, nije garancija ortogonalnosti.

**"Kasno filtriranje" (Finding 7) je kontraintuitivni pobednik.** Prirodno sam pomislio da agresivna ekstrakcija činjenica — Mem0 stil "iscedi u jednu čistu činjenicu" — mora da je pravi dizajn. Ablacija kaže suprotno. `Fast-Memorize` uništi `Fine-Memorize` na LoCoMo (25.5 vs 2.5 EM). Sirovi redovi pobeđuju sumarije na *sva četiri* metrika. Zašto? Zato što agresivna ekstrakcija baci ono vezivno tkivo koje kasnije čini multi-hop razmišljanje mogućim. Ti u trenutku upisa ne znaš koja će sitnica biti bitna u kombinaciji sa nečim kasnije. Pa — sačuvaj sada, filtriraj kasnije. Zato moja implementacija čuva sirove redove verbatim.

**Refleksija je overrated za pretragu.** M3 rezultat — SimpleMem `Planning+Reflect` radi *gori* od `Planning-Only` — je onaj koji bih najviše voleo da vidim replikovan čvrsto. Trenutni agent-building duh stavlja refleksiju/rethink svugde. Za memoriju barem, podaci kažu da dodaje trošak bez dobitka. Kad je ruta jednom planirana, dodatno razmišljanje oslabi odluku. Implementirao sam planning (query expansion) i namerno *nisam* dodao refleksiju.

## Šta je bilo teže nego što sam mislio

Embedingi bez modela. Svi sistemi iz papira koriste prave embeding modele. Meni je trebalo da radi na mašini bez API ključeva i bez skidanja modela, pa sam ispekao deterministički char-n-gram hash embeding. Dovoljno smislen da cosine radi na malom datasetu, ali daleko slabije od pravog enkodera — i to je najveći jaz između moje implementacije i sistema iz papira. Papir to i predviđa (O5, backbone robustnost): redosled dobrih naspram loših dizajna je prilično stabilan kroz backbonove, jer se uzemljenje (grounding) dešava pre generacije.

Konzervativna konsolidacija je frčka. "Spoji dve memorije kad su o istoj stvari" zvuči prosto. U praksi je prag delikatan: previše labav — srušiš različite činjenice; previše strog — nikad ne spojiš pa memorija raste neograničeno. Finding 9 ("konzervativni merge pobeđuje, delayed flush gubi") je lak za reći, ali stvarni prag (ja sam uzeo 0.85 cosine + entitet preklapanje) je pravi inženjerski knob.

Multi-versioning traži disciplinu. Logička invalidacija (nikad ne briši, samo označi kao zastarelo) je elegantna, ali znači da svaki upit mora po defaultu da filtrira `valid=True` — a *opciono* da uključi invalidne kad je upit eksplicitno temporalan ("gde sam *ranije* živeo?"). Taj dupli mod je lako pogrešiti.

## Graf cene naspram kvaliteta

Ovo je jedan od korisnijih dijagrama u celom radu, vredi zapamtiti — normalized utility naspram prosečnog kašnjenja po upitu:

```
utility
100 │                           · Cognee (84 @ 116s)
    │                    · Zep (84 @ 155s)
 80 │             · MemoryOS (82 @ 29s)
    │
 60 │       · A-MEM (58 @ 18s) · MemTree (64 @ 16s)
    │
 40 │ · LightMem (48 @ 4s)
    │
 20 │       · MemoChat (28 @ 15s) · Mem0 (21 @ 36s)
    └─────────────────────────────────────────────► latency (log)
       1s      10s        100s       1000s
```

Pareto fronta ide: **LightMem → MemTree → A-MEM → MemoryOS**, pa onda oštar skok do Cognee/Zep za poslednjih par poena utility-ja po 4-10× većoj ceni. Poruka za graditelja: izaberi tačku na toj krivoj prema tome koliko ti je workload osetljiv na latenciju. Najveći utility nije uvek vredan toga.

## Reference
- Papir: https://arxiv.org/abs/2606.24775
- Zvanični kod/benchmark: https://github.com/OpenDataBox/MemoryData
- Moja implementacija: `implementation/`
- Breakdown: `breakdown.md`
