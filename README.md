# BWINF 43 – Aufgabe 2: Flaggenmusterung

Dieses Projekt analysiert und vergleicht Flaggenbilder anhand von Farben, geometrischen Layouts, Seitenverhältnissen und Symbolen.

---

<img width="1552" height="234" alt="Screenshot 2026-11-22 at 14 56 17" src="https://github.com/user-attachments/assets/a6697e64-d397-4797-b1f8-cfb42061a991" />

---

## 1. Voraussetzungen

- **Python 3.9+**
- **pip** (Python-Paketmanager)

---

## 2. Installation der Abhängigkeiten

Es wird empfohlen, eine virtuelle Umgebung (`venv`) zu verwenden:

```bash
python3 -m venv venv
source venv/bin/activate       # Auf macOS / Linux
# bzw. unter Windows:
venv\Scripts\activate
```

Installieren Sie anschließend die benötigten Bibliotheken via `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Enthaltene Abhängigkeiten: `numpy`, `Pillow`)*

---

## 3. Ausführung / Erste Schritte

### Schritt 1: Vorverarbeitung der Flaggenbilder (Erforderlich)

Vor dem ersten Start der Vergleiche müssen die Rohbilder aus `Flaggen/` vorverarbeitet werden:

```bash
python3 preprocess.py
```

**Was dieser Schritt tut:**
- Skaliert alle Flaggen auf 100×100 Pixel und speichert sie im Ordner `Flaggen100/`.
- Extrahiert Merkmale (Farbverteilung, Kanten- und Quadranten-Abstraktion, Seitenverhältnisse, Sonderformen wie bei Nepal).
- Legt für jede Flagge eine `.txt`-Datei mit diesen Attributen in `Flaggen100/` an.

---
Beispiel - Israel.txt:

<img width="1664" height="832" alt="Screenshot 2026-11-22 at 14 56 57" src="https://github.com/user-attachments/assets/aef4376f-1c0e-45d2-8df4-8316f14a1d53" />

---

### Schritt 2: Ähnlichste Flaggen für ein bestimmtes Land ermitteln

Starten Sie die interaktive Suche mit:

```bash
python3 flaggen.py
```

Geben Sie an der Eingabeaufforderung das gesuchte Land ein (ohne Dateiendung), beispielsweise:
- `Deutschland`
- `Costa Rica`
- `Peru`
- `Nepal`

**Ausgabe:** Die 5 ähnlichsten Flaggen mit genauer Aufschlüsselung der Übereinstimmung (Farbe, Layout, Symbole, Gesamtübereinstimmung).

---
Beispiel - Ecuador:

<img width="1654" height="356" alt="Screenshot 2026-11-22 at 14 58 05" src="https://github.com/user-attachments/assets/6217c59b-3bc1-4d41-b034-593b28ec422d" />

---

### Schritt 3: Global ähnlichste Paare und Gruppen ermitteln

Um die 10 ähnlichsten Paare über den gesamten Datensatz sowie geclusterte Flaggengruppen zu berechnen:

```bash
python3 flaggenAll.py
```

**Ausgabe:**
- Die 10 ähnlichsten Flaggenpaare.
- Die Top 10 Gruppen zusammenhängender, ähnlicher Flaggen (Complete-Linkage Clustering).

---

<img width="1986" height="784" alt="Screenshot 2026-11-22 at 14 59 07" src="https://github.com/user-attachments/assets/266a5d39-c82a-4e98-99db-5f49d3954649" />

---

## 4. Dateistruktur

```
.
├── Flaggen/           # 200 vorgegebene Original-Flaggen (PNG)
├── Flaggen100/        # (Wird durch preprocess.py erzeugt) 100×100 Bilder & Metadaten (.txt)
├── flaggen.py         # Suche der Top 5 ähnlichsten Flaggen für ein bestimmtes Land
├── flaggenAll.py      # Ermittlung der 10 ähnlichsten Paare & Gruppen über alle Flaggen
├── preprocess.py      # Vorverarbeitung und Merkmalsextraktion
├── requirements.txt   # Python-Abhängigkeiten (numpy, Pillow)
├── submission.pdf     # Ausführliche Dokumentation und Lösung
└── README.md          # Markdown-Dokumentation
```
