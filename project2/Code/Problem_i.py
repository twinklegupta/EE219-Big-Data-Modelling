from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB    

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

#Extracting the specified categories
cat_four=['comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'misc.forsale',
'soc.religion.christian']
 
# Loading the data of all caegories
twenty_train = fetch_20newsgroups(subset='train',categories=cat_four, remove=('headers','footers','quotes'))
# Loading the data of all caegories
twenty_test =  fetch_20newsgroups(subset='test',categories=cat_four, remove=('headers','footers','quotes'))

# Send all documents to get cleaned/processed for analysis
size, = twenty_train.filenames.shape
count =0
while count < size :
    temp = twenty_train.data[count]
    twenty_train.data[count] = clean_data(temp)
    count+=1

# Send all documents to get cleaned/processed for analysis
size2, = twenty_test.filenames.shape
count =0
while count < size2 :
    temp = twenty_test.data[count]
    twenty_test.data[count] = clean_data(temp)
    count+=1

# create the TFxIDF vector representations for all docs from training set
count_vect = CountVectorizer()
train_vector = count_vect.fit_transform(twenty_train.data)

# Calculate the required txidf matrix by transorming the above formed training vector and get categoriesteh required dimensions
tf_transformer = TfidfTransformer(use_idf=True).fit(train_vector)
train_tfidf_matrix = tf_transformer.transform(train_vector)
docs1,terms1 =train_tfidf_matrix .shape


#Dimensionalisty reduction to 50
#Applying LSI,SVD for normalization
#for training data
svd = TruncatedSVD(n_components=50, random_state=42)
train_tfidf_svd = svd.fit_transform(train_tfidf_matrix )
train_tfidf_final = Normalizer(copy=False).fit_transform(train_tfidf_svd)
train_target_final = twenty_train.target
  

# create the TFxIDF vector representations for all docs from testing set
test_vector = count_vect.transform(twenty_test.data)

# Calculate the required txidf matrix by transorming the above formed testing vector and get categoriesteh required dimensions
tf_transformer = TfidfTransformer(use_idf=True).fit(test_vector)
test_tfidf_matrix = tf_transformer.transform(test_vector)
docs2,terms2 = test_tfidf_matrix.shape


#Dimensionalisty reduction to 50
#Applying LSI,SVD for normalization
#for testing data
test_tfidf_svd = svd.transform(test_tfidf_matrix)
test_tfidf_final = Normalizer(copy=False).fit_transform(test_tfidf_svd)
target_test = twenty_test.target

#Training and fitting the OneVsRestClassifier for Multiclass SVM
onevsrest = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_tfidf_final, train_target_final)
test_predicted1 = onevsrest.predict(test_tfidf_final)

#Calculating the accuracy, recall, precision and confusion matrix
#Multiclass SVM with One vs Resr Method:
accuracy1 = np.mean(target_test == test_predicted1)
print("Multiclass SVM with One vs Rest Method: \n")
print ('Current accuracy is :- ',accuracy1)
print ("Precision, recall values, f1-score,support are :-")
print (metrics.classification_report(target_test, test_predicted1))
print ('Current confusion matrix')
print (metrics.confusion_matrix(target_test, test_predicted1))
print("***************************************************************")

#Training and fitting the OneVsOneClassifier for Multiclass SVM
onevsone = OneVsOneClassifier(LinearSVC(random_state=0)).fit(train_tfidf_final, train_target_final)
test_predicted2 = onevsone.predict(test_tfidf_final)

#Calculating the accuracy, recall, precision and confusion matrix
#Multiclass SVM with One vs One Method:

accuracy2 = np.mean(target_test == test_predicted2)
print("Multiclass SVM with One vs One Method: \n")
print ('Current accuracy is :- ',accuracy2)
print ("Precision, recall values, f1-score,support are :-")
print (metrics.classification_report(target_test, test_predicted2))
print ('Current confusion matrix')
print (metrics.confusion_matrix(target_test, test_predicted2))
print("***************************************************************")

#Training and Fitting the Multi class Naive Bayes Model with OneVsRestClassifier
onevsrest = OneVsRestClassifier(GaussianNB()).fit(train_tfidf_final, train_target_final)
test_predicted3 = onevsrest.predict(test_tfidf_final)

#Calculating the accuracy, recall, precision and confusion matrix
#Multi class Naive Bayes Model with OneVsRes
accuracy3 = np.mean(target_test == test_predicted3)
print("Multi class Naive Bayes Model with OneVsRest \n")
print ('Current accuracy is :- ',accuracy3)
print ("Precision, recall values, f1-score,support are :-")
print (metrics.classification_report(target_test, test_predicted3))
print ('Current confusion matrix')
print (metrics.confusion_matrix(target_test, test_predicted3))

print("***************************************************************")
#Training and Fitting the Multi class Naive Bayes Model with OneVsOneClassifier
onevsone = OneVsOneClassifier(GaussianNB()).fit(train_tfidf_final, train_target_final)
test_predicted4 = onevsone.predict(test_tfidf_final)

#Calculating the accuracy, recall, precision and confusion matrix
#Multi class Naive Bayes Model with OneVsOneClassifier
accuracy4 = np.mean(target_test == test_predicted4)
print("Multi class Naive Bayes Model with OneVsOneClassifier  \n")
print ('Current accuracy is :- ',accuracy4)
print ("Precision, recall values, f1-score,support are :-")
print (metrics.classification_report(target_test, test_predicted4))
print ('Current confusion matrix')
print("***************************************************************")

print (metrics.confusion_matrix(target_test, test_predicted4))