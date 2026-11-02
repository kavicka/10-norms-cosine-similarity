# Tutoriál · 10 · Normy, kosínusová podobnosť a kedy kosínus = skalárny súčin

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál prejde [`cosine.py`](cosine.py) a [`bge_m3_check.py`](bge_m3_check.py) krok za krokom: čo každý experiment počíta, prečo je navrhnutý takto a čo si odniesť do nasledujúcich projektov o vyhľadávaní.

**Cieľ:** odpovedať na *„kedy je kosínusová podobnosť to isté ako skalárny súčin?“* (keď majú oba vektory dĺžku 1) a pochopiť, prečo záleží na tom, ktorú z nich vyhľadávanie používa.

**Potrebuješ:** projekt 03 (skalárny súčin, kosínus). NumPy a scikit-learn; na posledný krok lokálnu Ollamu s `ollama pull bge-m3`.

---

## Krok 1 · Tri spôsoby, ako merať dĺžku (`norms`)

```python
v = np.array([3.0, -4.0])
L1   = np.abs(v).sum()        # 7   „taxíková vzdialenosť“
L2   = np.linalg.norm(v)      # 5   priama dĺžka
Linf = np.abs(v).max()        # 4   najväčšia súradnica
```

**Čo je norma:** pravidlo, ktoré každému vektoru priradí dĺžku. Rôzne pravidlá, rôzne dĺžky pre ten istý vektor.

**Ako sa kreslia „jednotkové kružnice“:** zober 400 smerov dookola a každý naškáluj tak, aby mal normu presne 1 (`d / ‖d‖_p`). Pre L1 to vykreslí kosoštvorec, pre L2 kružnicu, pre L∞ štvorec (`assets/norms_high_dim.png`).

**Prečo na tom záleží:** embeddingy takmer vždy používajú **L2** a „normalizovať vektor“ znamená vydeliť ho jeho L2 normou, aby mal dĺžku 1.

---

## Krok 2 · Vysoké dimenzie sa správajú čudne (`high_dim`)

```python
for d in (2, 10, 100, 1000, 10000):
    X = rng.normal(size=(2000, d))
    norm = np.linalg.norm(X, axis=1)                    # dĺžky
    a, b = l2_normalize(X[:1000]), l2_normalize(X[1000:])
    cos = (a * b).sum(1)                                # kosínus 1000 náhodných dvojíc
```

**Čo sa meria a čo uvidíš:**
- **Dĺžky sa koncentrujú:** každý náhodný vektor má dĺžku ≈ √d a rozptyl sa zmenšuje (2,2 % pri d = 1000). Vo vysokých dimenziách majú teda náhodné vektory takmer rovnakú dĺžku.
- **Náhodné smery sú takmer kolmé:** smerodajná odchýlka kosínusu je ≈ 1/√d (0,0317 vs 0,0316 pri d = 1000) a 99,9 % náhodných dvojíc má |cos| < 0,1.

**Prečo na tom záleží pri vyhľadávaní:**
- V 1024-rozmernom priestore embeddingov sú „nesúvisiace“ texty blízko cos 0, takže aj stredný kosínus už nesie signál.
- Veľké rozdiely v dĺžke medzi skutočnými embeddingmi teda nie sú náhoda; pochádzajú z modelu alebo textu. Surový skalárny súčin by ich nechal ovládnuť poradie (Krok 4).

**Prečo `(a * b).sum(1)` a nie `a @ b.T`:** chceme kosínus 1000 *dvojíc* (riadok i s riadkom i), nie všetkých 1000 × 1000 kombinácií.

---

## Krok 3 · Identity, overené numericky (`identities`)

```
a · b = ‖a‖ ‖b‖ cos θ         →   cos θ = a·b / (‖a‖ ‖b‖)
ak ‖a‖ = ‖b‖ = 1:   a·b = cos θ   a   ‖a − b‖² = 2 − 2 cos θ
```

Druhý riadok vyjde z rozpísania `‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b`, keď sú obe normy rovné 1.

**Na jednotkových vektoroch** teda najväčší skalárny súčin = najväčší kosínus = najmenšia euklidovská vzdialenosť. Tri metriky, jedno poradie.

```python
D = rng.normal(size=(1000, 384)) * rng.uniform(0.2, 5, (1000, 1))   # zámerne náhodné dĺžky
Q = rng.normal(size=(200, 384))
dot = Q @ D.T
cos = l2_normalize(Q) @ l2_normalize(D).T
```

**Prečo zámerne náhodné dĺžky:** aby bolo vidieť, čo sa stane, keď embeddingy *nie sú* normalizované. 384 rozmerov je veľkosť malého modelu na embeddingy viet.

**Čo sa porovnáva:**
- identita `a·b = ‖a‖‖b‖cos θ`: max chyba 6,8e-13 (zaokrúhlenie),
- normalizovaný skalárny súčin vs kosínus: rozdiel 0,0,
- normalizovaná vzdialenosť² vs `2 − 2cos`: 1,3e-15,
- **top-10 susedov** pre každý z 200 dopytov: kosínus vs normalizovaný skalárny súčin a vs normalizovaná L2 sú zhodné pre 100 % dopytov; kosínus vs **surový** skalárny súčin pre **0 %** (len 39 % prekryv).

**Prečo surový skalárny súčin zlyhá:** jeho top-10 dokumentov má priemernú normu 84,9 oproti 50,4 celkovo. Odmeňuje *dĺžku*, nie len smer.

---

## Krok 4 · Pozri sa na efekt na slovenskom texte (`retrieval`)

```python
vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5))
D = vec.fit_transform(DOCS).toarray()
q = vec.transform([QUERY]).toarray()[0]
dot = D @ q
cos = dot / (np.linalg.norm(D, axis=1) * np.linalg.norm(q))
```

**Korpus:** päť krátkych poistných udalostí plus jeden dlhý všeobecný text poistných podmienok, ktorý opakuje slová ako *poškodenie*, *strecha*, *dom*. Dopyt: „krupobitie poškodilo strechu domu“.

**Prečo znakové 3–5-gramy (`char_wb`):** slovenčina skloňuje (*strecha / strechu / strechy*). Počty celých slov by ich brali ako rôzne slová; kúsky ako `str`, `stre`, `rech` sú spoločné, takže zhoda prežije. `char_wb` berie n-gramy len vo vnútri slov.

**Výsledok:**
- **Surový skalárny súčin:** vyhrá dlhý text podmienok (296 vs 76), jednoducho preto, že má všetkého viac (norma 72 vs ~11).
- **Kosínus:** dĺžku vydelí a na prvé miesto dá skutočnú udalosť s krupobitím na streche (0,782 vs 0,464).

**Prečo na tom neskôr záleží:** presne preto TF-IDF v scikit-learn štandardne normalizuje L2 a preto má BM25 (projekt 14 a ďalej) penalizáciu za dĺžku dokumentu.

---

## Krok 5 · Over skutočný model na embeddingy ([`bge_m3_check.py`](bge_m3_check.py))

```python
body = json.dumps({"model": "bge-m3", "input": TEXTS}).encode()
req = urllib.request.Request("http://localhost:11434/api/embed", body, {"Content-Type": "application/json"})
emb = json.loads(urllib.request.urlopen(req).read())["embeddings"]
norms = [math.sqrt(sum(x * x for x in e)) for e in emb]
```

**Prečo len štandardná knižnica:** jeden HTTP request na lokálny Ollama server, žiadna ďalšia závislosť.

**Prečo tieto štyri texty:** dve slovenské udalosti s krupobitím (mali by byť blízko), jedna anglická s krupobitím (test viacjazyčnosti) a recept na halušky (nesúvisí).

**Výsledok:**
- 1024 rozmerov, **každá norma = 1,0000**. bge-m3 normalizuje výstup, takže skalárny súčin = kosínus a rýchly index na skalárny súčin je bezpečný.
- SK–SK krupobitie 0,81, SK–EN krupobitie 0,66–0,77, recept 0,34–0,39.

**Všimni si dno:** „nesúvisiace“ je ~0,35, nie 0 ako náhodné vektory v Kroku 2. Skutočné embeddingy zdieľajú spoločný smer (nie sú izotropné). Skóre preto posudzuj **relatívne voči sebe**, nikdy podľa absolútneho prahu zvoleného od oka.

---

## Krok 6 · Ulož a nakresli

```bash
uv run python cosine.py        # → results.json
uv run python bge_m3_check.py  # → results_bge_m3.json (Mac s Ollamou)
uv run python plot.py          # → assets/*.png, cosine_vs_dot.gif
```

GIF ukazuje vektor dopytu `a` a dokument `b`: keď `b` predĺžiš, skalárny súčin stúpne, kým kosínus stojí; keď `b` otočíš, pohnú sa oba.

**Prečo nie kapitola v Racing Lab:** otázka „ktorý dokument je najbližšie k dopytu?“ patrí k textu, nie k autám.

---

## Vyskúšaj sám

1. Nahraď `CountVectorizer` za `TfidfVectorizer` (ten štandardne normalizuje L2). Zoradí teraz surový skalárny súčin rovnako ako kosínus?
2. V `identities` nastav všetky dĺžky na 1 pred výpočtom `dot`. Over, že z 0 % bude 100 %.
3. Pridaj do `bge_m3_check.py` piaty text o krupobití na aute. Kam sa zaradí medzi udalosti so strechou a recept?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 1 | Norma = dĺžka; embeddingy používajú L2; normalizovať = vydeliť L2 normou |
| 2 | Vysoké dimenzie: dĺžky sa koncentrujú pri √d, náhodné dvojice majú cos ≈ 0 |
| 3 | Na jednotkových vektoroch skalárny súčin = kosínus a vzdialenosť dá rovnaké poradie |
| 4 | Bez normalizácie skalárny súčin odmeňuje dlhé dokumenty |
| 5 | Over svoj model: bge-m3 je už normalizovaný; skutočné „nesúvisiace“ ≈ 0,35, nie 0 |
