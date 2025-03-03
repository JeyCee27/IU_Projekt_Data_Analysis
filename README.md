# Kurze Erklärung zur Nutzung des Codes


# Grundlegendes:
Die zu analysierende Datei muss im txt Format mit dem Code zusammen in einem Verzeichnis gespeichert werden.
Somit müssen die Dateien "Datasetprojpowerbi.txt" und "requirements.txt" zusammen mit dem Code "Textanalyse_03_03_2025" in einem Verzeichnis gespeichert werden
Die Abhängigkeiten können mit dem Terminal Befehl pip install -r requirements.txt  instaliert werden.

# Anpassen des Codes
Bei der Vektorizierung werden sowohl Unigrams (Einzelwörter) und Bigrams (Zwei Wöerter) verwendet. Wenn nur Unigrams verwendet werden sollen muss die "ngram_range" im code auf (1,1) geändert werden.
Bei der Extrahierung der Themen gibt "n_components" die Anzahl der Themen an die Etrahiert werden sollen 
Beim aufrufen der Funktion topic_extraction gibt "num_words" die Anzahl der Wörter pro Thema an.
Diese Werte können je nch Anforderung geändert werden.

# Nutzen einer andere Datei:
Wenn eine andere Datei als die hier Vorliegene Beispieldatei eingelesen werden soll, muss im Code der Name der zu öffnenden Datei geändert werden.

# Ausgabe der Zwischenergebnisse als Datei:
Um den mit den zwei Methoden Bow & Tfidf Vektorisierten Text detalierter zu betrachten, kann diser als Datei ausgegeben werden. Der dazu benötigte Befehl ist im Code in einem Kommentare zu finden.



