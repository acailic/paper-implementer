# Writeup — GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents

> Kako bih ovo objasnio drugu dok pijemo pivo — da me pita "sta citas?"

> **Jezici:** Engleski · [Srpski](writeup.md) (ovaj fajl)

---

U svijetu AI agenata ima debata koju svi imaju mišljenje o, a niko nije pravilno testirao: da li su kompjuterski agenti bolji preko grafičkog interfejsa (klikanje, tipkanje, prevlačenje po desktopu) ili preko programskih skillova (CLI komande koje manipulišu stanjem aplikacije)?

Svaki benchmark do sada je poređivao ove dvije modalitete sa različitim taskovima, različitim početnim uslovima, različitim kriterijumima uspjeha i različitim pravilima o tome šta agent smije raditi. To nije poređenje — to su dva odvojena eksperimenta koja se prave da su jedno.

Ovaj rad ispravlja to. Prave 440 desktop taskova kroz 18 pravih aplikacija — Audacity, LibreOffice, GIMP, draw.io, Chrome, Zoom, FreeCAD, i tako dalje — i provlače ih kroz i screen-only GUI agenta i skill-mediated CLI agenta sa potpuno istim opisom taska, istim početnim stanjem i istim final-state verifier-om. Jedina stvar koja se mijenja je akcijski interfejs. Screenshots i klikovi vs CLI-Anything skillovi.

## Glavni brojevi

Najjači GUI agent (GPT-5.4) dobija **59.1%** full pass rate. Najjači CLI agent sa originalnim skill-ovima (Codex GPT-5.5) dobija **48.2%**. GPT-5.4 je slabiji model ali pobjeđuje kroz GUI. To je već zanimljivo — interfejs može kompenzirati ograničenja modela.

Ali evo pravog pointa. Autori se pitaju: koliko od tog CLI gapa je zato što je skill sloj jednostavno nepotpun? Sistematski auditiraju svaki CLI-Anything skill protiv verifier checkpoint-a. Odgovor: **samo 37.6% verifier checkpoint-a može biti zadovoljeno originalnim skill-ovima**. Šezdeset-odsto nečeg što task zahtijeva jednostavno nije izloženo kroz CLI skill interfejs.

Zato rade dijagnostiku gdje zakrpaju skillove (koristeći verifier informacije — ovo je eksplicitno dijagnostički upper bound, ne deployable rješenje). CLI skače sa 48.2% na **69.3%**. Sada pobjeđuje najbolji GUI agent. CLI nije bio slabiji — njegovi alati su bili nepotpuni.

## Šta me najviše iznenadilo

Poredak po workflow-ovima izbacuje nekoliko intuicija. Pretpostavio sam da će GUI dominirati vizualnim taskovima tipa GIMP i draw.io jer su to "vizuelne" aplikacije. Pogrešno. CLI je kompetitivan ili čak jači u Visual Design zato što su draw.io taskovi zapravo o strukturiranih artefaktima — stranice, oblici, label-e, konektori — i oni se čisto mapiraju na programatske operacije. CLI agent kaže "dodaj oblik UserService na stranicu 2" direktno, dok GUI agent mora vizuelno navigirati canvas, postavljati oblike, tipkati labele, crtati konektore i pratiti stanje kroz više stranica bez da išta izgubi.

Gdje GUI zaista dominira: Audio (Audacity label trackovi), Prezentacije (LibreOffice Impress manipulacija slajdovima), Komunikacija (Zoom postavke). Ovo su slučajevi gdje je aplikacijski interfejs sam workflow — meni struktura i vidljivi kontrole direktno izlažu korake koje trebaš. GUI agent vidi putanju; CLI agent mora rekonstruirati je iz nepotpune skill dokumentacije.

Drugo iznenađenje: procedural grounding eksperiment. Kad daš GUI agentu eksplicitne korak-po-korak instrukcije ("klikni Tracks > Add New > Mono Track, zatim klikni dropdown strelicu, izaberi Name..."), full pass rate se jedva pomjera — sa 59.7% na 60.2%. Ali prosječno vrijeme izvršenja pada za 20%. Agent prestaje trošiti korake na eksploraciju. Ali i dalje ne uspijeva istom stopom jer je pravi bottleneck vizuelni grounding — pouzdano klikanje prave stvari, praćenje stanja kroz duge sekvence, ne odustajanje prerano.

## Failure modovi su potpuno različiti

CLI agenti ne uspijevaju zato što:
- Skill sloj ne izlaže operaciju koju trebaju (skill coverage gap)
- Moraju pogoditi default-e koje GUI korisnici automatski nasljeđuju — konvencije imenovanja objekata, interni identifikatori, label vs name distinkcije
- Kritično stanje aplikacije nije vidljivo kroz nijedan skill, pa haluciniraju vjerovatno-ali-pogrešne konfiguracije

GUI agenti ne uspijevaju zato što:
- Ne mogu pronaći pravu kontrolu — menije, tabovi, dijaloge, skrivene postavke. Klikaju oko sebe tražeći pravu putanju i istroše korake.
- Pogreše workflow — pogrešan redoslijed operacija, nedostajuće potvrdni dijalozi, prestaju prerano.
- Deklariraju uspjeh bez provjere. Prodju kroz plauzibilnu sekvencu akcija i kažu GOTovo, ali izvezeni fajl ne postoji ili sačuvano stanje se ustvari nije promijenilo.

Ovo su zaista komplementarno. CLI-jev bottleneck je širina i tačnost njegovog tool interfejsa. GUI-jev bottleneck je pouzdanost perceptualno-motornog lanca izvršenja.

## Što ovo znači za ljude koji grade agente

Tri zaključka koje bih ponio sa sobom:

**Skill coverage je centralni scaling problem za CLI agente.** 37.6% coverage nije mala rupa — to je dominantno objašnjenje CLI underperformance-a. Ako gradiš skill-mediated agenta, kvalitet i coverage tvog skill sloja je važniji od izbora modela. Rad ne rješava ovo (automatsko gradjenje skillova sa visokim coverage-om ostaje otvoreno) ali kvantificira tačno koliko je u pitanju.

**Izbor modaliteta treba biti workflow-dependent, ne app-category-dependent.** "Vizuelne" aplikacije automatski ne favorizuju GUI. Ono što je bitno je da li interfejs direktno izlaže planirani workflow (GUI pobjeđuje) ili je ciljno stanje strukturirani artefakt (CLI pobjeđuje). Adaptivni router koji bira modalitet po task-u bi pobijedio oboje samog.

**GUI-CLI pitanje je ustvari o tome gdje živi izvršna logika.** U GUI-u, aplikacijski interfejs enkodira workflow — agent ga otkriva. U CLI-u, skill sloj enkodira workflow — agent ga poziva. Nijedno nije inherentno bolje. Pravi dizajn pitanje je: gdje treba biti inženjerisana izvršna struktura taska? U vidljivim workflow-ima, verifikovanim skill interfejsima, ili hibridnim okruženjima koji kombinuju oboje?

## Reference
- Rad: https://arxiv.org/abs/2606.24551
- Službeni kod: https://github.com/rebeccaz4/gui-vs-cli
- Breakdown: `breakdown.md`
