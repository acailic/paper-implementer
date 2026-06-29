# Writeup — ShutterMuse: Capture-Time Photography Guidance with MLLMs

> Evo kako bih ovo objasnio prijatelju pre piva, ako me pita "šta si čitao?"

> **Jezici:** [English](writeup.md) · Srpski (ovaj fajl)

---

Zamisli da fotografiš zamak. Kadraš, ali nešto nije u redu. Horizont je nagnut, zamak je sićušan u ćošku, a u kadru je kontejner. Šta radiš? Obrišeš? Ostaviš kako jeste? Baciš sve i pokušaš ponovo?

A sada zamisli da si ti *subjekt* — tvoj drug te fotografiše na nekim stepenicama i ne zna kako da te postavi. Gde da staviš ruke? Kako da nagneš telo?

Svaki postojeći AI alat za fotografiju pretpostavlja opciju jedan: "evo ti fotke, naćiću ti bolji crop." Ali to je često pogrešno. Ponekad je kadar već odličan. Ponekad je fotka neopravljiva. I niko nije pravio alate koji kažu *osobi na fotki* kako da stoji.

Ovaj rad popravlja sva tri problema odjednom.

## Trostruka odluka

Centralna ideja zvuči skoro trivijalno u retrospektivi: **ne treba svaku fotku cropovati.** Autori definišu tri odluke:

- **Refine** — snimak ima potencijal, ali kadar treba prilagoditi (crop/rekompozicija)
- **Keep** — kadar je već dobar, ostavi ga na miru
- **Reject** — fotka je bez nade (blur, nema subjekta, ekstremni nagib)

Svaki prethodni benchmark je pretpostavljao da svaka slika ima preferirani crop. Specijalizovani modeli za cropovanje kao Venus i InstructCrop su bukvalno nesposobni da kažu "ne cropuj ovo" — uvek izbace bounding box, čak i kad to stvari pogorša. Na benchmarku ti modeli score-uju **nula** na i reject i keep success rate. To nije mala mana — to je fundamentalni mismatch između onoga za šta su alati napravljeni i onoga što fotografima zapravo treba.

## Šta su izgradili

Tri artefakta, složena jedan na drugi:

**CaptureGuide-Bench** — benchmark sa 421 photographer-side primera (trostruka odluka sa 3–5 ekspertskih bounding box-ova po refine) i 552 subject-side primera (preporuke poze uslovljene scenom). Evaluira se i geometrijskim merama (IoU, BDE) i MLLM sudijom koji boduje kompozicioni kvalitet.

**CaptureGuide-Dataset** — ~130K trening primera. Photographer strana je izgrađena kroz expert-seeded, MLLM-verified self-distillation pipeline (EMDP): 10 anotatora pravi 12K seed primera, MLLM ih normalizuje, model pseudo-labira 500K nenalabeliranih slika, i verifier (Gemini-3.0-Pro) proverava kvalitet pre prihvatanja. Tri runda proširuju dataset na 100K. Subject strana koristi pametan trik: kreneš od portretnih fotki, ukloniš osobu image editing modelom, izvučeš pozu keypoint-ima pomoću YOLO-a, i generiše scene-conditional rationale pomoću drugog MLLM-a. 30K primera.

**ShutterMuse** — unified MLLM na Qwen3-VL-8B koji radi oba zadatka. Prva faza je supervised fine-tuning na strukturiranim JSON izlazima. Druga faza je GRPO (reinforcement learning) sa tri reward komponente: tačna odluka, očuvanje subjekta u crop box-u, i konzistentnost vidljivosti za poze.

## Dve stvari koje su me stvarno iznenadile

Prvo — **keep/reject jaz je dramatičan, a nitko nije to primijetio.** Specijalizovani modeli za cropovanje koje papers slave kao state-of-the-art (Venus, InstructCrop) bukvalno score-uju 0% na reject success rate i skoro 0% na keep success rate. To znači da im daš savršeno komponovanu fotku — oni je iscropuju. Daš im neupotrebljivo zamućenu papkaru — oni i dalje pokušavaju da je "poboljšaju". Generalni MLLM-i kao GPT-5.5 i Gemini su bolji u odluci (48-51% refinement rate), ali im je crop box daleko (IoU ~65% naspram ShutterMuse-ovih 74%). Niko ranije nije bio dobar na oba fronta.

Drugo — **8B model može da parira GPT-Image-2 za preporučivanje poze pri 20× većoj brzini.** Na subject-side preporuci poze, Nano-Banana-Pro vodi sa mean score 0.39, GPT-Image-2 dobija 0.35, a ShutterMuse 0.34. Kvalitetni jaz je mali. Ali ShutterMuse to radi za 5 sekundi sa 412 tokena, naspram 55-103 sekunde i 1300-1400 tokena za foundation modele. Za real-time "stani ovako" asistenta ugrađenog u kameru aplikaciju, to je razlika između korisnog i nekorisnog.

## Šta ablation kaže

GRPO faza je pravi game-changer. Sam SFT te dovede do IoU 72.39% i RSR 68.97%. Dodaj GRPO sa punim reward-om i skačeš na IoU 74.30%, RSR 82.76%, KSR 74.55%. Decision reward (`Rdec`) je najvažnija pojedinačna komponenta — ukloniš je i RSR pada sa 83% na 62%, a KSR sa 75% na 65%. Ima smisla: trostruka odluka je novi dio, i RL reward ga direktno trenira.

Mask preservation reward (`Rmask`) je pametan. Koristi BiRefNet da detektuje glavni subjekat, pa proverava da li predikcioni crop box pokriva ≥90% maske subjekta. To sprečava da model nauči da "varaju" cropovanjem u prazan prostor. Bez njega, MLLM-Score pada, što potvrđuje da zaista pomaže kompozicionom kvalitetu.

## Šta je bilo teže nego što se očekivalo

Dataset pipeline je pravi inženjerski podvig. Self-distillation petlja — treniraj model → pseudo-label → MLLM verifikuj → re-treniraj — se lako može zavrtiti u garbage-in-garbage-out. Kontrolišu to sa fiksnim expert validation set-om koji prati kvalitet u svakoj rundi, nezavisnim expert test set-om kojeg pipeline nikad ne dodiruje, i verifier-om koji održava >87% F1 kroz sve kategorije. Podaci se šire sa 12K na 100K bez degradacije kvaliteta, što je impresivno za pipeline sa toliko pokretnih delova.

Subject-side konstrukcija je posebno kreativna. Kreni od postojećih portretnih fotki, ukloni osobu, i koristi izvučenu pozu kao "odgovor" za sada praznu scenu — pametan način da bootstrap-uješ dataset za problem koji bi inače zahtijevao stotinke photoshoot-a sa anotacijama poza.

## User study vrijedi spomenuti

MLLM-Score i human preference ranking se slažu sa SRCC = 0.90 na photographer strani i bukvalno identično na subject strani. To je jako dobar dokaz da evaluacioni protokol nije samo artefakt sudskog modela već hvata stvarnu perceptivnu razliku.

## Šta nedostaje

Kvalitetni jaz na subject strani (0.34 vs 0.39) je stvaran. COCO-17 keypoint format ne može da reprezentuje kontakt stopala — Appendix D pokazuje lebdeće noge u skeleton vizualizaciji, priznato kao ograničenje. Model radi samo sa single frame-om, ne video. I trening podaci vjerovatno skew-uju prema konvencionalnim kompozicionim normama — bilo bi zanimljivo vidjeti kako se ovo generalizuje kroz različite fotografske tradicije.

## Reference

- Papir: https://arxiv.org/abs/2606.25763
- Zvanični kod: https://github.com/lijayuTnT/ShutterMuse
- Project page: https://lijayutnt.github.io/ShutterMuse/
- Breakdown: `breakdown.md`
