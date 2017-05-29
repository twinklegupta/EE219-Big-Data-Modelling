import re

from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import nltk.stem
from nltk.stem.porter import PorterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer

import numpy as np


stemmer2 = nltk.stem.SnowballStemmer('english')
stop_words = text.ENGLISH_STOP_WORDS #stopwords
# The function removes some stemming words like go , going
# Also atakes care of punctuation marks, stop words etc
# Basically cleans the Data
def clean_data(temp):
    temp = re.sub("[,.-:/()]"," ",temp)
    temp = temp.lower()
    words = temp.split()
    after_stop=[w for w in words if not w in stop_words]
    after_stem=[stemmer2.stem(plura1) for plura1 in after_stop]
    temp=" ".join(after_stem)
    return temp

categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

comp_rec = ['comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey']

# Loading the data of all caegories
twenty_train = fetch_20newsgroups(subset='train',categories=comp_rec,  remove=('headers','footers','quotes'))

# Send all documents to get cleaned/processed for analysis
size, = twenty_train.filenames.shape
count =0
while count < size :
    temp = twenty_train.data[count]
    twenty_train.data[count] = clean_data(temp)
    count+=1

# create the TFxIDF vector representations for all docs

count_vect = CountVectorizer()
twenty_td_vector = count_vect.fit_transform(twenty_train.data)

# Calculate the required txidf matrix by transorming the above formed vector and get categoriesteh required dimensions
tf_transformer = TfidfTransformer(use_idf=True).fit(twenty_td_vector)
twenty_tfidf_matrix = tf_transformer.transform(twenty_td_vector)
doc_count,term_count = twenty_tfidf_matrix.shape
# 
# Apply LSI to the TFxIDF matrix
# and map each document to a 50-dimensional vector.
#Performing dimensionality reduction using LSI and SVD
svd = TruncatedSVD(n_components=50, n_iter=10,random_state=42)
twenty_tfidf_matrix_svd = svd.fit_transform(twenty_tfidf_matrix)
twenty_tfidf_matrix_normalize = Normalizer(copy=False).fit_transform(twenty_tfidf_matrix_svd)
doc_count,term_count = twenty_tfidf_matrix_normalize.shape
print ('After applying LSI normalized txidf matrix size :- ',term_count)