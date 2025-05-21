import csv
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
nltk.download('all')

#Laden der zu analysierenden Datei
with open('Datasetprojpowerbi.txt', 'r') as file:
    reader = csv.reader(file)
    df = [row[1] for row in reader]

#Definieren der Stopwörter
stopwords = stopwords.words("english")
add_stopwords = ("made", "ve", "i've", "ix", "i'll", "feel")
stopwords.extend(add_stopwords)

#Textvorverarbeitung mit Lemmatisation
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = stopwords
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmatized_tokens)
#Vektorisierung mit BoW
count_vect = CountVectorizer(preprocessor=preprocess_text, analyzer='word',
                       ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None,)
bow = count_vect.fit_transform(df)
bow_df = pd.DataFrame(bow.toarray(), columns=count_vect.get_feature_names_out())

print('Bag of word Vokabular:')
print(bow_df)

#Mit dem folgenden Code kann der mit BoW vektorisierte Text als Datei ausgegeben werden.
#bow.to_csv('bow_Vokabular.csv', index=False)

#Vektorisierung mit Tf-Idf
tfidf_vect = TfidfVectorizer(preprocessor=preprocess_text, analyzer='word',
                       ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None,)
tfidf = tfidf_vect.fit_transform(df)
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tfidf_vect.get_feature_names_out())

print('TfIdf Vokabular:')
print(tfidf_df)

#Mit dem folgenden Code kann der mit Tfidf vektorisierte Text als Datei ausgegeben werden.
#tfidf.to_csv('tfidf_Vokabular.csv', index=False)


#Themen Extrahieren
def topic_extraction(model, feature_names, num_words, topic_list):
    for i, topic in enumerate(model.components_):
        top_n = [feature_names[i]
                 for i in topic.argsort()[:-num_words - 1:-1]]
        topics = ' '.join(top_n)
        topic_list.append(f"Topic {i}: {topics}")
        print(f"Topic {i}: {topics}")
    print()

#Optimale Anzahl an Themen finden mit LDA

#Maximale Anzahl an zu testenden Themen definieren
test_topic = 25

# Testen mit LDA und Bow
lda_model = LatentDirichletAllocation(n_components=test_topic, learning_method='online', random_state=42, max_iter=20)
lda =lda_model.fit_transform(bow)


sse = []
for k in range(1, test_topic):
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init='auto', max_iter=1000)
    kmeans = kmeans.fit(lda)
    sse.append(kmeans.inertia_)

# Daten visulaisieren
plt.plot(range(1, test_topic), sse)
plt.xticks(range(1, test_topic))
plt.xlabel("Anzahl an Themen")
plt.ylabel("SSE")
plt.show()

# Parameter Definieren:
# Beste Anzahl an Themen
n_components = 9
#Anzahl an Wörtern pro Thema
num_words = 10

# LSA mit BoW:
lsa_bow = TruncatedSVD(n_components=n_components, algorithm='randomized',n_iter=20, random_state=42)
lsa = lsa_bow.fit_transform(bow_df)
terms = count_vect.get_feature_names_out()

lsa_bow_topics = []
print('Themen mit LSA und Bag of words:')
topic_extraction(lsa_bow, terms, num_words, lsa_bow_topics)


#LSA mit TfIdf
lsa_tfidf = TruncatedSVD(n_components=n_components, algorithm='randomized',n_iter=20, random_state=42)
lsa = lsa_tfidf.fit_transform(tfidf_df)
terms = tfidf_vect.get_feature_names_out()

lsa_tfidf_topis = []
print('Themen mit LSA und Tf-Idf:')
topic_extraction(lsa_tfidf, terms, num_words, lsa_tfidf_topis)


#LDA mit BoW:
lda_bow = LatentDirichletAllocation(n_components=n_components, learning_method='online', random_state=42, max_iter=20)
lda =lda_bow.fit_transform(bow_df)
terms = count_vect.get_feature_names_out()

lda_bow_topics = []
print('Themen mit LDA und Bag of words:')
topic_extraction(lda_bow, terms, num_words, lda_bow_topics)


#LDA mit TFIDF:
lda_tfidf = LatentDirichletAllocation(n_components=n_components, learning_method='online', random_state=42, max_iter=20)
lda =lda_tfidf.fit_transform(tfidf_df)
terms = tfidf_vect.get_feature_names_out()

lda_tfidf_topis = []
print('Themen mit LDA und Tf-Idf:')
topic_extraction(lda_tfidf, terms, num_words, lda_tfidf_topis)
