import csv
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('all')

#Laden der zu analysierenden Datei
with open('Datasetprojpowerbi.txt', 'r') as file:
    reader = csv.reader(file)
    df = [row[1] for row in reader]

stopwords = stopwords.words("english")
add_stopwords =("made", "ve", "i've", "ix", "i'll")
stopwords.extend(add_stopwords)

#Text vorverarbeitung mit Lemmatisation
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = stopwords
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmatized_tokens)

#Vektorizierung mit BoW
count_vect = CountVectorizer(preprocessor=preprocess_text, analyzer='word',
                       ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None,)
bow = count_vect.fit_transform(df)
bow = pd.DataFrame(bow.toarray(), columns=count_vect.get_feature_names_out())

print('Bag of word Vokabular:')
print(bow)

#Mit dem folgenden Code kann der mit BoW vektorizierte Text als Datei ausgegeben werden.
#bow.to_csv('bow_Vokabular.csv', index=False)

#Vektorizierung mit Tf-Idf
tfidf_vect = TfidfVectorizer(preprocessor=preprocess_text, analyzer='word',
                       ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None,)
tfidf = tfidf_vect.fit_transform(df)
tfidf = pd.DataFrame(tfidf.toarray(), columns=tfidf_vect.get_feature_names_out())

print('TfIdf Vokabular:')
print(tfidf)

#Mit dem folgenden Code kann der mit Tfidf vektorizierte Text als Datei ausgegeben werden.
#tfidf.to_csv('tfidf_Vokabular.csv', index=False)


#Themen Extrahieren
def topic_extraction(model, feature_names, num_words):
    for i, topic in enumerate(model.components_):
        result = "Topic "+str(i)+ ": "
        result += ", ".join([feature_names[i]
                             for i in topic.argsort()[:-num_words - 1:-1]])
        print(result)
    print()


#LSA mit BoW:
lsa_bow = TruncatedSVD(n_components=20, algorithm='randomized',n_iter=10, random_state=42)
lsa = lsa_bow.fit_transform(bow)
terms = count_vect.get_feature_names_out()

print('Themen mit LSA und Bag of words:')
topic_extraction(lsa_bow, terms, 10,)

#LSA mit TfIdf
lsa_tfidf = TruncatedSVD(n_components=20, algorithm='randomized',n_iter=10, random_state=42)
lsa = lsa_tfidf.fit_transform(tfidf)
terms = tfidf_vect.get_feature_names_out()

print('Themen mit LSA und Tf-Idf:')
topic_extraction(lsa_tfidf, terms, 10,)


#LDA it BoW:
lda_bow = LatentDirichletAllocation(n_components=20, learning_method='online', random_state=42, max_iter=10)
lda =lda_bow.fit_transform(bow)
terms = count_vect.get_feature_names_out()

print('Themen mit LDA und Bag of words:')
topic_extraction(lda_bow, terms, 10,)


#LDA mit TFIDF:
lda_tfidf = LatentDirichletAllocation(n_components=20, learning_method='online', random_state=42, max_iter=10)
lda =lda_tfidf.fit_transform(tfidf)
terms = tfidf_vect.get_feature_names_out()

print('Themen mit LDA und Tf-Idf:')
topic_extraction(lda_tfidf, terms, 10,)

