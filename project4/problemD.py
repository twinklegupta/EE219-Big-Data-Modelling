import re

from nltk.corpus import stopwords
import nltk.stem
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import numpy as np
import scipy
from sklearn.decomposition import TruncatedSVD, NMF, PCA, SparsePCA
from sklearn.metrics import confusion_matrix, cluster
import pandas as pd
import matplotlib.pyplot as plt


stemmer = nltk.stem.SnowballStemmer('english')
stop_words = text.ENGLISH_STOP_WORDS  # stopwords


def clean_data(temp):
    
    temp = temp.lower()
    
    # This statement will remove all the punctuation marks
    tokenize1 = RegexpTokenizer(r'\w+')
    
    # Stemmer is used to stem the words, eg. -ing removal
    stemming = PorterStemmer()
    
    # The tokenize function creates tokens from the text
    token = tokenize1.tokenize(temp)
    
    # This statement removes all the stop words such as articles that do not contribute towards the meaning of the text
    no_stops = [x for x in token if not x in stop_words]
    
    # This statement will perform the actual stemming
    no_ing = [stemming.stem(y) for y in no_stops]
    results = [stemmer.stem(y) for y in no_ing]
    
    # This statement removes all the digits
    results = [z for z in results if not z.isdigit()]
    
    # Each word is appended with a space before it
    return " ".join(results)





categories = [
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey'
              ]

# Loading the data of all categories
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True,
                                  remove=('headers', 'footers'))

labels = twenty_train.target

print("labels :- ", labels.shape)

# Get length of training data


len_twenty_data, = twenty_train.filenames.shape

# Passing every document for processing.
count = 0
while count < len_twenty_data:
    temp = twenty_train.data[count]
    twenty_train.data[count] = clean_data(temp)
    count += 1

# create the TFxIDF vector representations for all docs

count_vect = CountVectorizer()
twenty_td_vector = count_vect.fit_transform(twenty_train.data)

# Calculate the required txidf matrix by transorming the above formed vector and get categoriesteh required dimensions
tf_transformer = TfidfTransformer(use_idf=True).fit(twenty_td_vector)
twenty_tfidf_matrix = tf_transformer.transform(twenty_td_vector)
doc_count, term_count = twenty_tfidf_matrix.shape
# print(twenty_tfidf_matrix)

print ("Total number of terms are", term_count)


nmf = NMF(n_components=4,init = 'random', random_state=0)
twenty_tfidf_matrix_r1 = nmf.fit_transform(twenty_tfidf_matrix)
twenty_tfidf_matrix_r1= np.log(np.add(twenty_tfidf_matrix_r1,1.0))
km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, precompute_distances='auto', verbose=0,
            copy_x=True, n_jobs=1, algorithm='auto').fit(twenty_tfidf_matrix_r1)

pca = PCA(n_components=2)
twenty_tfidf_matrix_r = pca.fit_transform(twenty_tfidf_matrix_r1)

count1 = 0
labels1 = labels//4
confusion = confusion_matrix(labels1,km.labels_)
print("Confusion")
print(confusion)
print("Homogeneity:" ,metrics.homogeneity_score(labels1, km.labels_))
print("Adjusted Rand-Index: " ,metrics.adjusted_rand_score(labels1, km.labels_))
print("Completeness: " ,metrics.completeness_score(labels1, km.labels_))
print("V-measure: " ,metrics.v_measure_score(labels1, km.labels_))
print("Mutual Information: ", metrics.adjusted_mutual_info_score(labels1, km.labels_))
print("confusion : ", confusion_matrix(labels1,km.labels_ ))
count = 0
for l in km.labels_:
    if labels1[count] == 0 and l == 0 :
        labels1[count] = 4
    elif labels1[count] == 1 and l == 0 :
        labels1[count] = 1
    elif labels1[count] == 0 and l == 1 :
        labels1[count] = 2
    elif labels1[count] == 1 and l == 1 :
        labels1[count] = 3
    count = count+1

x,y = twenty_tfidf_matrix_r[:,0], twenty_tfidf_matrix_r[:,1]
df = pd.DataFrame(dict(x=x, y=y, label=labels1))

groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.x, group.y, marker='.', linestyle='', label=name)
ax.legend()
