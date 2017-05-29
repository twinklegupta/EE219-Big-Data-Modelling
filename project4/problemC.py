#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:59:23 2017

@author: twinklegupta
"""


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
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.metrics import confusion_matrix, cluster
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt

stemmer = nltk.stem.SnowballStemmer('english')
stop_words = text.ENGLISH_STOP_WORDS  # stopwords


def clean_data(temp):

    temp = temp.lower()
    tokenize1 = RegexpTokenizer(r'\w+')
    stemming = PorterStemmer()
    token = tokenize1.tokenize(temp)
    no_stops = [x for x in token if not x in stop_words]
    no_ing = [stemming.stem(y) for y in no_stops]
    results = [stemmer.stem(y) for y in no_ing]
    results = [z for z in results if not z.isdigit()]
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
labels1 = labels//4
print("labels :- ", labels.shape)

len_twenty_data, = twenty_train.filenames.shape

count = 0
while count < len_twenty_data:
    temp = twenty_train.data[count]
    twenty_train.data[count] = clean_data(temp)
    count += 1
    
count_vect = CountVectorizer()
twenty_td_vector = count_vect.fit_transform(twenty_train.data)

# Calculate the required txidf matrix by transorming the above formed vector and get categoriesteh required dimensions
tf_transformer = TfidfTransformer(use_idf=True).fit(twenty_td_vector)
twenty_tfidf_matrix = tf_transformer.transform(twenty_td_vector)
doc_count, term_count = twenty_tfidf_matrix.shape

print ("Total number of terms are", term_count)

################################### SVD #####################################
max_purity = -1
res_dim = 0

for dim in range(2,20):
    svd = TruncatedSVD(n_components=dim ,n_iter=10, random_state=42)
    svd.fit(twenty_tfidf_matrix)
    twenty_tfidf_matrix_r=svd.fit_transform(twenty_tfidf_matrix)
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, precompute_distances='auto', verbose=0,
                copy_x=True, n_jobs=1, algorithm='auto').fit(twenty_tfidf_matrix_r)
    
    clustered_labels = []
    c = 0
    for i in labels:
        if (i < 4):
            clustered_labels.append(1)
        else:
            clustered_labels.append(0)
            c += 1
    clustered_labels1 = np.array(clustered_labels)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Dimension :  ",dim)
    print("Homogeneity:" ,metrics.homogeneity_score(labels1, km.labels_))
    print("Adjusted Rand-Index: " ,metrics.adjusted_rand_score(labels1, km.labels_))
    print("Completeness: " ,metrics.completeness_score(labels1, km.labels_))
    print("V-measure: " ,metrics.v_measure_score(labels1, km.labels_))
    
    print("Mutual Information: %0.3f", metrics.adjusted_mutual_info_score(clustered_labels, km.labels_))
    print("Explained variance of the SVD step with {} components: {}%".format(dim, int(explained_variance * 100)))
    print("confusion : ", confusion_matrix(labels1,km.labels_ ))
    if max_purity < metrics.adjusted_rand_score(labels1, km.labels_) + metrics.homogeneity_score(labels1, km.labels_):
        max_purity = metrics.adjusted_rand_score(labels1, km.labels_) + metrics.homogeneity_score(labels1, km.labels_)
        res_dim = dim
    
print(res_dim)
    
## resultant dimension found to be 15


################################### NMF without normalise ####################

max_purity = -1
res_dim = 0

for dim in range(2,20):
    nmf = NMF(n_components=dim ,init = 'random', random_state=0)
    #nmf.fit(twenty_tfidf_matrix)
    twenty_tfidf_matrix_r=nmf.fit_transform(twenty_tfidf_matrix)
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, precompute_distances='auto', verbose=0,
                copy_x=True, n_jobs=1, algorithm='auto').fit(twenty_tfidf_matrix_r)
    
    clustered_labels = []
    c = 0
    for i in labels:
        if (i < 4):
            clustered_labels.append(1)
        else:
            clustered_labels.append(0)
            c += 1
    clustered_labels1 = np.array(clustered_labels)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Dimension :  ",dim)
    print("Homogeneity:" ,metrics.homogeneity_score(labels1, km.labels_))
    print("Adjusted Rand-Index: " ,metrics.adjusted_rand_score(labels1, km.labels_))
    print("Completeness: " ,metrics.completeness_score(labels1, km.labels_))
    print("V-measure: " ,metrics.v_measure_score(labels1, km.labels_))
    print("Mutual Information: ", metrics.adjusted_mutual_info_score(labels1, km.labels_))
    print("Explained variance of the NMF step with {} components: {}%".format(dim, int(explained_variance * 100)))
    print("confusion : ", confusion_matrix(labels1,km.labels_))
    if max_purity < metrics.adjusted_rand_score(labels1, km.labels_) + metrics.homogeneity_score(labels1, km.labels_):
        max_purity = metrics.adjusted_rand_score(labels1, km.labels_) + metrics.homogeneity_score(labels1, km.labels_)
        res_dim = dim
    
print(res_dim)
    
## resultant dimension found to be 4 but ran d score values were really low, so had to normalise

################################### NMF with normalise ####################

max_purity = -1
res_dim = 0

for dim in range(2,20):
    nmf = NMF(n_components=dim ,init = 'random', random_state=0)
    #nmf.fit(twenty_tfidf_matrix)
    twenty_tfidf_matrix_r=nmf.fit_transform(twenty_tfidf_matrix)
    twenty_tfidf_matrix_r = normalize(twenty_tfidf_matrix_r, norm='l2', axis = 1,copy=True)
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, precompute_distances='auto', verbose=0,
                copy_x=True, n_jobs=1, algorithm='auto').fit(twenty_tfidf_matrix_r)
    
    clustered_labels = []
    c = 0
    for i in labels:
        if (i < 4):
            clustered_labels.append(1)
        else:
            clustered_labels.append(0)
            c += 1
    clustered_labels1 = np.array(clustered_labels)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Dimension with normalization:  ",dim)
    print("Homogeneity:" ,metrics.homogeneity_score(labels1, km.labels_))
    print("Adjusted Rand-Index: " ,metrics.adjusted_rand_score(labels1, km.labels_))
    print("Completeness: " ,metrics.completeness_score(labels1, km.labels_))
    print("V-measure: " ,metrics.v_measure_score(labels1, km.labels_))
    print("Mutual Information: ", metrics.adjusted_mutual_info_score(labels1, km.labels_))
    print("Explained variance of the NMF step with {} components: {}%".format(dim, int(explained_variance * 100)))
    print("confusion : ", confusion_matrix(labels1,km.labels_))
    if max_purity < metrics.adjusted_rand_score(labels1, km.labels_) + metrics.homogeneity_score(labels1, km.labels_):
        max_purity = metrics.adjusted_rand_score(labels1, km.labels_) + metrics.homogeneity_score(labels1, km.labels_)
        res_dim = dim
    
print(res_dim)
#    
### resultant dimension found to be 3


####################### Plot data points #####################################

svd = NMF(n_components=2).fit(twenty_tfidf_matrix)
twenty_tfidf_matrix_r = svd.fit_transform(twenty_tfidf_matrix)

x,y = twenty_tfidf_matrix_r[:,0], twenty_tfidf_matrix_r[:,1]

df = pd.DataFrame(dict(x=x, y=y, label=1))

groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.x, group.y, marker='.', linestyle='', label=name)
ax.legend()

#################### Plot with log ###########################################
svd = NMF(n_components=2).fit(twenty_tfidf_matrix)
twenty_tfidf_matrix_r = svd.fit_transform(twenty_tfidf_matrix)

x,y = twenty_tfidf_matrix_r[:,0],np.log(twenty_tfidf_matrix_r[:,1])

df = pd.DataFrame(dict(x=x, y=y, label=1))

groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.x, group.y, marker='.', linestyle='', label=name)
ax.legend()
