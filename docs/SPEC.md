# Projektspezifikation: Fehlererkennung im 3D-Druck

Stand: Oct 4, 2026 · @Leon

## Ziel und Kontext

Gebaut wird ein kleines, sauber strukturiertes Python-Projekt, das Fehler im FDM-3D-Druck auf Bildern erkennt. Es ist ein Bewerbungsprojekt und soll auf GitHub oeffentlich sichtbar sein.

Hintergrund: Ich studiere Maschinenbau im Master und bewerbe mich auf Stellen, deren Profil Informatik oder Robotik, gute Python-Kenntnisse, Softwareengineering-Grundlagen (APIs, Datenstrukturen, Testen) und Interesse an Machine-Learning-Systemen oder Computer Vision verlangt. Das Projekt soll genau diese Punkte belegen.

Der eigentliche Wert liegt nicht in einem hohen Genauigkeitswert, sondern in sauberer Methodik, nachvollziehbarer Struktur und begruendeten Entscheidungen. Ein Leser soll das Repository oeffnen und in fuenf Minuten erkennen koennen, wie ich arbeite.

Zielgruppe des Repositories sind technische Recruiter und Entwickler, die den Code ueberfliegen, das README lesen und die Commit-Historie ansehen.

## Fachlicher Hintergrund

Beim FDM-Druck (Schmelzschichtung) wird Kunststoff durch eine beheizte Duese Schicht fuer Schicht aufgetragen. Stimmen Flussrate, Duesenabstand, Temperatur oder Haftung nicht, entstehen charakteristische Fehlerbilder, die auf Fotos gut sichtbar sind.

| Fehler | Aussehen | Typische Ursache |
| --- | --- | --- |
| Spaghetti | Wirres Filamentknaeuel ueber dem Bauteil | Bauteil hat sich geloest, Druck laeuft ins Leere |
| Warping | Hochgebogene Ecken, Ablösen vom Druckbett | Schrumpfung beim Abkuehlen, schlechte Betthaftung |
| Stringing | Duenne Faeden zwischen Bauteilbereichen | Zu hohe Temperatur, fehlender Rueckzug |
| Zits / Blobs | Punktuelle Materialanhaeufungen | Druckschwankungen beim Start und Stopp der Extrusion |
| Under extrusion | Luecken und duenne Schichten | Zu geringe Flussrate, verstopfte Duese |

Diese Fehler treten mitten im Druck auf und bleiben oft stundenlang unbemerkt. Eine Kamera mit Bilderkennung kann sie frueh melden, Material sparen und spaeter sogar automatisch pausieren. Das macht die Aufgabe praxisnah und erklaerbar.

Mein Maschinenbau-Hintergrund ist hier ein Vorteil: Ich kann im Bewerbungsgespraech begruenden, warum ein Fehlerbild so aussieht, nicht nur dass ein Modell es erkennt.

## Datengrundlage

Die Bilddaten stammen von Roboflow Universe (universe.roboflow.com), einer Plattform mit nutzergenerierten, bereits beschrifteten Bilddatensaetzen. Ich habe keinen Zugang zu einem 3D-Drucker und erzeuge keine eigenen Aufnahmen.

Ausgewaehlt wird ein Datensatz zum Thema 3D-Druckfehler nach diesen Kriterien:

1. Mindestens etwa 300 Bilder, ideal 500 bis 2000.
2. Zwei bis vier klar unterscheidbare Klassen, zum Beispiel spaghetti, stringing, warping.
3. Keine Klasse mit weniger als rund 30 Beispielen; solche Klassen werden zusammengelegt oder entfernt.
4. Lizenz MIT oder CC BY 4.0.

Download im **YOLOv8-Format** ueber das Python-Paket `roboflow`. Der Datensatz besteht dann aus `train/`, `valid/`, `test/` mit je `images/` und `labels/` sowie einer `data.yaml`, die Pfade und Klassennamen enthaelt. Jede Label-Datei enthaelt pro Objekt eine Zeile `klassen_id x_center y_center width height` mit auf 0 bis 1 normierten Werten.

Harte Regeln zum Umgang mit den Daten:

- Keine Bilddaten und keine Modellgewichte ins Git-Repository. Der Ordner `data/` und `*.pt` sind in der `.gitignore` ausgeschlossen.
- Der API-Schluessel steht ausschliesslich in `.env`, die nie committet wird. Im Repository liegt nur `.env.example` ohne echten Schluessel.
- Datensatzname, URL, Lizenz und BibTeX-Zitat gehoeren ins README.
- Hoechstens drei bis fuenf Beispielbilder duerfen in `docs/` liegen, um Ergebnisse zu illustrieren.

## Technischer Rahmen

| Punkt | Festlegung |
| --- | --- |
| Betriebssystem | Windows |
| Editor | PyCharm, Terminal integriert |
| Python | 3.x in einer venv unter `.venv` |
| Paketverwaltung | pip mit `requirements.txt` |
| Modellbibliothek | Ultralytics YOLOv8, Start mit `yolov8n.pt` |
| Weitere Pakete | roboflow, pyyaml, python-dotenv, pytest |
| Training | Google Colab mit kostenloser GPU; lokal nur Inferenz und Auswertung |
| Versionsverwaltung | Git, Remote auf GitHub, Hauptzweig `main` |

Wichtig zum Kenntnisstand: Ich bin **Anfaenger in Python und Machine Learning**. Ich kann Skripte lesen und anpassen, habe aber noch keine groesseren Projekte gebaut und kenne gaengige Werkzeuge wie pytest, argparse oder FastAPI nur oberflaechlich.

Daraus folgt fuer die Umsetzung: lieber wenige, kurze und gut kommentierte Module als eine clevere Abstraktion. Jede Datei soll fuer sich lesbar sein. Externe Bibliotheken nur, wenn sie wirklich noetig sind.

## Aktueller Stand

Das Repository heisst `detection` und liegt lokal unter `C:\Users\Leon\PycharmProjects\detection`, verbunden mit einem GitHub-Repository.

Bereits vorhanden:

- Leere Ordner `data/`, `docs/`, `notebooks/`
- `src/detection/` mit `__init__.py` und einer Beispieldatei `preprocessing.py`
- `tests/` mit `test_preprocessing.py`
- `.gitignore` (Python-Vorlage von GitHub), `LICENSE` (MIT), `README.md`, `pytest.ini`, `requirements.txt`
- Eine virtuelle Umgebung unter `.venv`

`preprocessing.py` und `test_preprocessing.py` waren nur ein Platzhalterbeispiel und koennen geloescht oder ersetzt werden.

Noch nicht vorhanden: die Ordner `configs/` und `scripts/`, saemtliche Module aus dem naechsten Abschnitt, `.env` und `.env.example`, die projektspezifischen Ergaenzungen der `.gitignore` sowie ein ausgefuelltes README.

## Soll-Struktur

```
detection/
├── api/
│   ├── main.py                FastAPI-Anwendung
│   └── schemas.py             Pydantic-Modelle fuer Anfrage und Antwort
├── configs/
│   └── baseline.yaml          Einstellungen eines Experiments
├── data/                      Datensatz, nicht im Repository
│   └── .gitkeep
├── docs/                      Abbildungen, Ergebnisbilder, Notizen
├── notebooks/                 Exploration und Ergebnisdarstellung
├── reports/                   Erzeugte Auswertungen, nicht im Repository
├── scripts/
│   └── download_data.py       Datensatz von Roboflow holen
├── src/detection/
│   ├── __init__.py
│   ├── config.py              YAML laden und pruefen
│   ├── dataset_check.py       Datenpruefung
│   ├── grouping.py            Duplikate und Bildgruppen finden
│   ├── resplit.py             Gruppenweise Neuaufteilung
│   ├── train.py               Training starten
│   ├── evaluate.py            Auswertung und Fehleranalyse
│   └── predict.py             Ein Bild rein, Erkennungen raus
├── tests/                     Ein Test-Modul je Quellmodul
├── .env                       Geheimnisse, nie committen
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── pytest.ini
├── README.md
└── requirements.txt
```

`src` wird in PyCharm als Sources Root markiert, damit `from detection.config import load_config` funktioniert. Dieselbe Rolle erfuellt `pythonpath = src` in der `pytest.ini`.

Trennung der Verantwortlichkeiten: In `src/detection/` steht importierbare, getestete Logik. In `scripts/` stehen duenne Startskripte ohne eigene Logik. Notebooks importieren aus `src`, sie enthalten keine eigene Logik.

## Module und ihre Aufgaben

Alle Module bekommen Typangaben, deutsche oder englische Docstrings und mindestens einen pytest-Test.

**`config.py`** — Laedt eine YAML-Datei in eine `Config`-Dataclass mit den Feldern `name`, `data_yaml`, `model`, `epochs`, `imgsz`, `seed`. Prueft die Werte und wirft `ValueError` bei unsinnigen Angaben, `FileNotFoundError` bei fehlender Datei. Keine festen Werte im Code, alles kommt aus der Konfiguration.

**`scripts/download_data.py`** — Liest `ROBOFLOW_API_KEY` ueber python-dotenv aus `.env`, laedt den Datensatz im YOLOv8-Format nach `data/dataset/`. Verstaendliche Fehlermeldung, wenn der Schluessel fehlt. Workspace, Projekt und Version sind oben im Skript als Konstanten gesetzt.

**`dataset_check.py`** — Die Datenpruefung, der wichtigste eigenstaendige Beitrag des Projekts. Sie liefert:

1. Anzahl der Bilder je Teilmenge (train, valid, test)
2. Haeufigkeit jeder Klasse je Teilmenge, auch als Diagramm
3. Bilder ohne Label und Labels ohne Bild
4. Ungueltige Labels: Werte ausserhalb 0 bis 1, unbekannte Klassen-IDs, kaputte Zeilen
5. Verteilung der Bildgroessen und der Boxgroessen
6. Einen Textbericht nach `reports/dataset_report.md`

**`grouping.py`** — Findet nahezu identische Bilder ueber Wahrnehmungs-Hashes (`imagehash`, Verfahren `phash`) und fasst sie zu Gruppen zusammen. Hintergrund: Viele Roboflow-Datensaetze bestehen aus Einzelbildern von Videos. Zusaetzlich wird geprueft, ob solche Gruppen ueber train, valid und test verteilt sind. Das Ergebnis ist eine Zahl: wie viele Testbilder haben einen nahen Zwilling im Training.

**`resplit.py`** — Teilt den Datensatz gruppenweise neu auf (etwa 70/15/15), sodass eine Bildgruppe immer vollstaendig in genau einer Teilmenge landet. Schreibt die neue Aufteilung nach `data/dataset_clean/` und erzeugt eine passende `data.yaml`. Deterministisch ueber einen Seed.

**`train.py`** — Startet ein YOLO-Training anhand einer Konfigurationsdatei, aufgerufen ueber `python -m detection.train --config configs/baseline.yaml`. Nutzt argparse. Ergebnisse landen unter `runs/<name>/`. Der Seed wird gesetzt und mitprotokolliert.

**`evaluate.py`** — Bewertet ein trainiertes Modell auf der Testmenge. Gibt mAP50, mAP50-95 sowie Precision und Recall je Klasse aus, schreibt eine Konfusionsmatrix und eine Precision-Recall-Kurve nach `reports/`, und legt die zwanzig schlechtesten Vorhersagen als annotierte Bilder ab.

**`predict.py`** — Die Anwendung fuer ein einzelnes Bild, aufgerufen ueber `python -m detection.predict --image pfad/zum/bild.jpg --model runs/baseline/weights/best.pt`. Laedt das trainierte Modell, erkennt Fehler und gibt sie als JSON mit Klasse, Konfidenz und Bounding-Box aus. Speichert zusaetzlich das Bild mit eingezeichneten Rahmen nach `reports/predictions/`. Die Konfidenzschwelle ist als Option einstellbar, der Standardwert wird aus der PR-Kurve begruendet.

**Fest eingeplant, in einer spaeteren Etappe: `api/`** — Ein FastAPI-Dienst mit einem Endpunkt, der ein Bild entgegennimmt und erkannte Fehler als JSON zurueckgibt, dazu ein `/health`-Endpunkt. Eingabevalidierung ueber Pydantic, Tests ueber den FastAPI-TestClient, Verpackung in Docker.

Die API hat zwei Endpunkte: `POST /predict` nimmt eine Bilddatei entgegen und gibt dieselbe JSON-Struktur zurueck wie `predict.py`, `GET /health` meldet, ob der Dienst laeuft. Die Antwortform wird ueber Pydantic-Modelle festgelegt. Ungueltige Eingaben, also falscher Dateityp, leere Datei oder zu grosses Bild, ergeben saubere HTTP-Fehler statt Abstuerze. Das Modell wird einmal beim Start geladen, nicht pro Anfrage. Getestet wird mit dem FastAPI-TestClient, je ein Test fuer den Erfolgsfall und fuer jeden Fehlerfall. Dazu ein Dockerfile, damit der Dienst mit einem Befehl startet.

## Methodische Anforderung

Diese Punkte sind der inhaltliche Kern des Projekts und wichtiger als jede Modellkennzahl.

**Gruppenweise Aufteilung statt zufaelliger.** Der groesste Fallstrick: Viele Roboflow-Datensaetze enthalten Einzelbilder aus Videos. Aufeinanderfolgende Bilder sind fast identisch. Verteilt man sie zufaellig, sieht das Modell im Test dieselben Szenen wie im Training, die Kennzahlen sind hervorragend und wertlos. Deshalb werden Bildgruppen gebildet und vollstaendig einer Teilmenge zugeordnet.

**Beide Ergebnisse berichten.** Einmal mit der Originalaufteilung von Roboflow, einmal mit der sauberen Aufteilung. Der Unterschied ist das zentrale Ergebnis des Projekts und gehoert ins README.

**Metriken.** Reine Genauigkeit ist bei schiefer Klassenverteilung irrefuehrend. Berichtet werden mAP50 und mAP50-95 insgesamt und je Klasse, Precision und Recall je Klasse, eine Konfusionsmatrix und eine Precision-Recall-Kurve. Dazu eine kurze Diskussion, warum ein uebersehener Fehler teurer ist als ein Fehlalarm, und wie die Konfidenzschwelle entsprechend gewaehlt wird.

**Reproduzierbarkeit.** Jeder Lauf schreibt seine Konfiguration, den Seed und die Paketversionen in den Ergebnisordner. Gleiche Konfiguration plus gleicher Seed ergibt dasselbe Ergebnis.

**Streuung statt Einzelwert.** Wo es die Rechenzeit erlaubt, jeden Vergleich mit drei Seeds laufen lassen und Mittelwert plus Streuung angeben.

**Fehleranalyse statt nur Zahlen.** Die zwanzig schlechtesten Vorhersagen werden als Bilder gespeichert und im README kommentiert: Was ging schief und warum.

**Klassenungleichgewicht.** Die Klassenhaeufigkeiten werden ausgewiesen. Klassen mit sehr wenigen Beispielen werden zusammengelegt oder entfernt, und die Entscheidung wird begruendet.

## Qualitaetsanforderungen an den Code

- Typangaben an allen oeffentlichen Funktionen, Docstrings im Google-Stil mit Args und Returns.
- Funktionen kurz und mit einer klaren Aufgabe; keine Logik auf Modulebene ausser hinter `if __name__ == "__main__":`.
- Pfade immer ueber `pathlib.Path`, nie als zusammengesetzte Zeichenketten.
- Verstaendliche Fehlermeldungen mit `ValueError` und `FileNotFoundError` statt stiller Rueckgabe von `None`.
- Ein Test-Modul je Quellmodul. Tests nutzen `tmp_path` und erzeugen ihre Daten selbst, damit sie ohne Datensatz laufen.
- Jede Funktion bekommt mindestens einen Test fuer den Normalfall und einen fuer den Fehlerfall.
- Keine Geheimnisse im Repository. `.env` steht in der `.gitignore`, `.env.example` enthaelt nur Platzhalter.
- Formatierung und Pruefung mit Ruff; eine `pyproject.toml` oder `ruff.toml` mit Zeilenlaenge 100.
- Optional, wenn es ohne grossen Aufwand geht: ein GitHub-Actions-Workflow, der bei jedem Push `ruff check` und `pytest` ausfuehrt.

Git-Konventionen: kleine, thematisch abgegrenzte Commits mit aussagekraeftigen deutschen Nachrichten im Imperativ, zum Beispiel `Fuege Duplikatserkennung ueber phash hinzu`. Keine Sammelcommits wie `Update`. Groessere Themen in eigenen Branches, danach zusammenfuehren.

## Umsetzung in Etappen

Jede Etappe endet mit laufenden Tests und einem Commit. Erst wenn eine Etappe steht, beginnt die naechste.

1. **Geruest** — Ordner `configs/`, `scripts/`, `reports/` anlegen, `requirements.txt` fuellen, `.gitignore` ergaenzen, `.env.example` schreiben, `config.py` mit Tests. Ergebnis: `pytest` laeuft gruen durch.
2. **Daten holen** — `download_data.py` schreiben, Datensatz nach `data/dataset/` laden, Struktur und `data.yaml` pruefen. Ergebnis: Der Datensatz liegt lokal, nichts davon im Repository.
3. **Datenpruefung** — `dataset_check.py` mit allen sechs Pruefungen und dem Bericht unter `reports/dataset_report.md`.
4. **Gruppen und Neuaufteilung** — `grouping.py` und `resplit.py`. Ergebnis: eine Zahl dazu, wie viele Testbilder einen nahen Zwilling im Training haben, und ein sauber aufgeteilter Datensatz unter `data/dataset_clean/`.
5. **Erstes Training** — `train.py` mit `configs/baseline.yaml`, ausgefuehrt in Google Colab. Ergebnis: ein trainiertes Modell und die Standardausgaben von Ultralytics.
6. **Auswertung** — `evaluate.py` mit Kennzahlen, Konfusionsmatrix, PR-Kurve und den zwanzig schlechtesten Vorhersagen.
7. **Der Vergleich** — Dasselbe Training auf der Originalaufteilung und auf der sauberen Aufteilung, Ergebnisse gegenuebergestellt. Das ist die Kernaussage des Projekts.
8. **Anwendung** — `predict.py`: ein Bild hinein, Erkennungen als JSON und als markiertes Bild heraus. Ergebnis: Das Projekt ist von aussen benutzbar.
9. **API** — FastAPI-Dienst mit `POST /predict` und `GET /health`, Pydantic-Modelle, Tests fuer Erfolgs- und Fehlerfaelle, Dockerfile.
10. **README** — Vollstaendig ausfuellen mit Ergebnissen, Abbildungen, Begruendungen und einem Beispielaufruf der API.
11. **Optional: Modellvergleich** — yolov8n gegen yolov8s gegen yolov8m, Genauigkeit gegen Rechenzeit.

## README und Abnahme

Das README ist das wichtigste Einzelstueck des Projekts, weil die meisten Leser nur es lesen. Es braucht diese Abschnitte:

1. Titel und ein bis zwei Saetze, was das Projekt tut
2. Motivation: warum Fehlererkennung im 3D-Druck
3. Datensatz mit Quelle, Lizenz und Zitat
4. Projektstruktur als kurzer Verzeichnisbaum
5. Installation und Ausfuehrung, Befehl fuer Befehl zum Nachmachen
6. Ergebnisse: Kennzahlentabelle, Konfusionsmatrix, Beispielbilder
7. Der Aufteilungsvergleich als eigener, hervorgehobener Abschnitt
8. Entwurfsentscheidungen: warum YOLOv8, warum diese Metriken, warum diese Aufteilung
9. Grenzen und was ich anders machen wuerde
10. Lizenzhinweis

Das Projekt gilt als fertig, wenn alle Punkte zutreffen:

- [ ] `pytest` laeuft ohne Fehler durch
- [ ] `ruff check` meldet nichts
- [ ] Ein frischer Clone laesst sich allein nach dem README zum Laufen bringen
- [ ] Weder Daten noch Modellgewichte noch `.env` liegen im Repository
- [ ] `predict.py` liefert fuer ein beliebiges Testbild JSON und ein markiertes Bild
- [ ] Die API laeuft lokal, `POST /predict` gibt dieselbe Antwort wie `predict.py`, und ungueltige Eingaben ergeben saubere Fehler
- [ ] Das README enthaelt echte, selbst gemessene Zahlen und mindestens zwei Abbildungen
- [ ] Der Aufteilungsvergleich ist dokumentiert und mit Zahlen belegt
- [ ] Mindestens fuenfzehn nachvollziehbare Commits in der Historie

## Arbeitsweise

An Claude Code gerichtet:

- Arbeite die Etappen **einzeln** ab und halte nach jeder an. Baue nicht alles auf einmal.
- Erklaere vor jeder Datei in zwei bis drei Saetzen, was sie tut und warum sie so aufgebaut ist. Ich bin Anfaenger und will den Code verstehen, nicht nur besitzen.
- Bevorzuge einfache, lesbare Loesungen gegenueber eleganten Abstraktionen. Keine Metaklassen, keine Fabrikfunktionen, keine vorzeitige Verallgemeinerung.
- Frage nach, statt zu raten, besonders bei der Wahl des Roboflow-Datensatzes, den Klassennamen und dem Verhaeltnis der Aufteilung.
- Nenne nach jeder Etappe den Befehl zum Pruefen und einen Vorschlag fuer die Commit-Nachricht.
- Schreibe keine Platzhalterwerte in Ergebnisse. Kennzahlen entstehen nur aus echten Laeufen.

Nicht-Ziele, bewusst ausgeklammert:

- Keine eigene Datenerfassung, kein Zugriff auf einen Drucker, keine Anbindung an OctoPrint oder Moonraker
- Keine eigene Netzarchitektur; vortrainierte YOLO-Modelle genuegen
- Keine Echtzeitverarbeitung von Videostroemen
- Keine Benutzeroberflaeche ausser der optionalen API
- Kein Produktivbetrieb und keine Weiterverwertung; es ist ein Lern- und Bewerbungsprojekt
