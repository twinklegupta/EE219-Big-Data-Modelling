from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression

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
comp_rec = ['comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey']
 
# Loading the data of all caegories
twenty_train = fetch_20newsgroups(subset='train',categories=comp_rec, remove=('headers','footers','quotes'))
# Loading the data of all caegories
twenty_test =  fetch_20newsgroups(subset='test',categories=comp_rec, remove=('headers','footers','quotes'))

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
for i in range(0,len(train_target_final)):
    if(train_target_final[i] <= 3):
        train_target_final[i] = 0
    else:
        train_target_final[i] = 1    

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

shape, = twenty_test.target.shape
count =0
while count <shape :
    if(twenty_test.target[count] <= 3):
        twenty_test.target[count] = 0
    else:
        twenty_test.target[count] = 1    
    count+=1


logi_reg = LogisticRegression()
#fit the model using logistic regression
logi_reg.fit(train_tfidf_final, train_target_final)
#predicted values in Y_pred_logi
test_predicted = logi_reg.predict(test_tfidf_final)

#Calculating the accuracy, recall, precision and confusion matrix
accuracy_svm = np.mean(twenty_test.target == test_predicted)
print ('Current model Accuracy :-  ', accuracy_svm)
print ('\'0\' is Computer Technology and \'1\' is Recreational Activity')
print ("Precision, recall values, f1-score,support are :-")
print (metrics.classification_report(twenty_test.target,test_predicted))
print ('Current confusion matrix :- ')
print (metrics.confusion_matrix(twenty_test.target, test_predicted))

#Plotting the ROC
probas_ = logi_reg.predict_proba(test_tfidf_final)                                    
fpr, tpr, thresholds = roc_curve(twenty_test.target, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "SVM ROC")                                    
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive ')
plt.ylabel('True Positive ')
plt.title('Sample ROC')
plt.legend(loc="center")
plt.show()