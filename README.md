# Kurze Erklärung zur Nutzung des Codes


# Grundlegendes:
Die Dateien "Datasetprojpowerbi.txt" und "requirements.txt",die im txt Format vorliegen, müssen zusammen mit dem Code "Textanalyse_11_05_2025" in einem Verzeichnis gespeichert.
Die Abhängigkeiten können mit dem Terminal Befehl pip install -r requirements.txt  instaliert werden.

# Nutzung des Codes:
Nach dem Ausführen des Codes wird ein Diagramm angezeigt. Mit der dort abgebildeten Kurve lässt sich die Optimale Themenanzahl bestimmen. Dazu muss die Zahl auf der X-Achse abgelesen werden an der Stelle, wo die abgebildete Funktion biet bzw. einen knick hat. Bei der Beispieldatei ist dies an der Zahl 9. Nach dem ablesen der Zahl kann diese unter dem Kommentar „Parameter Definieren“ für n_components eingetragen werden.

# Anpassen des Codes
Wenn die Liste der Stopwörter erweitert werden soll, können diese der Liste add-stopwords hinzugefügt werden. 
Bei der Vektorisierung werden sowohl Unigrams (Einzelwörter) und Bigrams (Zwei Wöerter) verwendet. Wenn nur Unigrams verwendet werden sollen muss die "ngram_range" im code auf (1,1) geändert werden.

# Nutzen einer andere Datei:
Wenn eine andere Datei als die hier vorliegene Beispieldatei eingelesen werden soll, muss im Code der Name der zu öffnenden Datei geändert werden.

# Ausgabe der Zwischenergebnisse als Datei:
Um den mit den zwei Methoden Bow & Tfidf vektorisierten Text detailierter zu betrachten, kann dieser als Datei ausgegeben werden. Der dazu benötigte Befehl ist im Code in einem Kommentar zu finden.



