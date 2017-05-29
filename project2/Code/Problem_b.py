from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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

# Get length of training data
len_twenty_data, = twenty_train.filenames.shape

#Passing every document for processing.
count=0
while count <len_twenty_data :
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

#Reporting the final number of terms
print ("Total number of terms are", term_count)
