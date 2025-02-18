import csv
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('all')


#Datei laden
with open('Datasetprojpowerbi.txt', 'r') as file:
    reader = csv.reader(file)
    df = [row[1] for row in reader]

#Beschwerden Datei
pd.DataFrame(df).to_csv('Beschwerden.csv', index=False)

stopwords = ['the','is','be','hadn', 'some', 'in', 'each', "you've", 'ourselves', 'into', 'are', "haven't", "hasn't", "shouldn't", 'isn', 'so',
             'other', 'off', 'couldn', 'the', "it's", 'shan',"you'll", 'for', "mustn't", 'need', 'neddn' 'again', 'from', 'yourself',"you'd",
             'further', 'very', 'above', 'theirs', 'has', 'but', 'being', 'while', 'be', 'they', "didn't", "hadn't", 'most', 'a','out', 'been',
             'here', 'those', 'you', 'mightn', 've', 'don', "won't", "she's", 'i', "weren't", 'over', 'can','until', 'such', 'by', 'down', 'your',
             'o', 'on', 'all', 'aren', 'there', 'it', 'own', 'he', 'below', 'm', 'him', 'what', 'more', 'at','few', 'through', 'than', 'wouldn',
             'after', 'when', 'my', 'that', 'why', 'himself', 'll', 'had', 'should', 'with', 'wasn', 'too', 'which','against', 'me', "don't", 'hasn',
             'she', "needn't", 'y', 'if', 'this', 'because', 'any', 'of', 't', 'about', "aren't", 'our', "mightn't",'now', 'then', 'was', 'during',
             "should've", 'am', 'doing', 'is', 'yours', 'will', 'them', "doesn't", 'weren', 're', 'once', 'before','where', 'whom', 'these', 'doesn',
             'an', 'myself', 'how', "wasn't", 'hers', "shan't", 'does', 'mustn', 'haven', 'as', 'just','yourselves', 'ain', 'didn', 'd', 'do', 'ours',
             'his', 'herself', 'who', 'under', 'both', 'to', 'same', 'itself', 'her', 's', 'their','or', 'were', 'won', 'and', 'between', 'nor',
             'only', 'shouldn', "couldn't", 'its', 'have', "you're", 'having', 'themselves', 'we',"that'll", 'ma', 'did', 'up', 'would', 'give', 'us',
             'could', 'might', 'must', 'need', 'sha', 'wo', 'very' ]

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
                       ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None,)
bow = count_vect.fit_transform(df)
bow = pd.DataFrame(bow.toarray(), columns=count_vect.get_feature_names_out())

print('Bag of word Vokabular:')
print(bow)


#Datei erstellen
bow.to_csv('bow_Vokabular.csv', index=False)

#Vektorizierung mit Tf-Idf
tfidf_vect = TfidfVectorizer(preprocessor=preprocess_text, analyzer='word',
                       ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None,)
tfidf = tfidf_vect.fit_transform(df)
tfidf = pd.DataFrame(tfidf.toarray(), columns=tfidf_vect.get_feature_names_out())

print('TfIdf Vokabular:')
print(tfidf)


#Datei erstellen

tfidf.to_csv('tfidf_Vokabular.csv', index=False)


#Themen Extrahieren
def topic_extraction(model, feature_names, num_words):
    for i, topic in enumerate(model.components_):
        result = "Topic "+str(i)+ ": "
        result += ", ".join([feature_names[i]
                             for i in topic.argsort()[:-num_words - 1:-1]])
        print(result)
    print()


#LSA mit BoW:
lsa_bow = TruncatedSVD(n_components=10, algorithm='randomized',n_iter=10)
lsa = lsa_bow.fit_transform(bow)
terms = count_vect.get_feature_names_out()

print('Themen mit LSA und Bag of words:')
topic_extraction(lsa_bow, terms,10,)

#LSA mit TfIdf
lsa_tfidf = TruncatedSVD(n_components=10, algorithm='randomized',n_iter=10)
lsa = lsa_tfidf.fit_transform(tfidf)
terms = tfidf_vect.get_feature_names_out()

print('Themen mit LSA und Tf-Idf:')
topic_extraction(lsa_tfidf, terms,10,)

#LDA it BoW:
lda_bow = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=42, max_iter=1)
lda =lda_bow.fit_transform(bow)
terms = count_vect.get_feature_names_out()

print('Themen mit LDA und Bag of words:')
topic_extraction(lda_bow, terms, 10,)


#LDA mit TFIDF:
lda_tfidf = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=42, max_iter=1)
lda =lda_tfidf.fit_transform(tfidf)
terms = tfidf_vect.get_feature_names_out()

print('Themen mit LDA und Tf-Idf:')
topic_extraction(lda_tfidf, terms, 10,)

