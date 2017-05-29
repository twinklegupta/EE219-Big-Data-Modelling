from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
import nltk.stem
import re
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

# Sort tge tficf matrix and return 10 biggest/smost significant terms
def sort_matrix(row, features):
    row = row[0]
    yx = zip(row, features)
    #yx.sort()
    

    yx_sorted = sorted(yx, key=lambda yx: yx[0])


    print (yx_sorted[len(yx_sorted)-10 : len(yx_sorted)])
    return

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

# Loading the data of all caegories
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, remove=('headers','footers','quotes'))

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

#Creates an empty TF-ICF matrix
tf_icf = np.zeros(shape=(20,term_count))

# Create a tdm from teh tficf for every curr_category

count =0 
while count < doc_count:
    curr_category = twenty_train.target[count]
    tf_icf[curr_category] = tf_icf[curr_category,] + twenty_td_vector[count,]
    count+=1



#Calculates the TF-ICF for every category
tf_transformer = TfidfTransformer(use_idf=True).fit(twenty_td_vector)
tf_icf_final = tf_transformer.transform(tf_icf)
features = count_vect.get_feature_names()

output_categories = [3,4,6,15]

count =0 
while count < 4:
    print ('Ten most significant terms for ',categories[output_categories[count]], ' are  :- ')
    sort_matrix(tf_icf_final[output_categories[count]].toarray(), features)
    print('\n')
    count+=1
