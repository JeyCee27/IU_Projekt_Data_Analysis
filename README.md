# Kurze Erklärung zur Nutzung des Codes


# Grundlegendes:
Die Dateien "Datasetprojpowerbi.txt" und "requirements.txt",die im txt Format vorliegen, müssen zusammen mit dem Code "Textanalyse_11_05_2025" in einem Verzeichnis gespeichert.
Die Abhängigkeiten können mit dem Terminal Befehl pip install -r requirements.txt  instaliert werden.

# Anpassen des Codes
Bei der Vektorisierung werden sowohl Unigrams (Einzelwörter) und Bigrams (Zwei Wöerter) verwendet. Wenn nur Unigrams verwendet werden sollen muss die "ngram_range" im code auf (1,1) geändert werden.
Bei der Extrahierung der Themen gibt "n_components" die Anzahl der Themen an die Extrahiert werden sollen.
Beim Aufrufen der Funktion topic_extraction gibt "num_words" die Anzahl der Wörter pro Thema an.

# Nutzen einer andere Datei:
Wenn eine andere Datei als die hier vorliegene Beispieldatei eingelesen werden soll, muss im Code der Name der zu öffnenden Datei geändert werden.

# Ausgabe der Zwischenergebnisse als Datei:
Um den mit den zwei Methoden Bow & Tfidf vektorisierten Text detailierter zu betrachten, kann dieser als Datei ausgegeben werden. Der dazu benötigte Befehl ist im Code in einem Kommentar zu finden.



