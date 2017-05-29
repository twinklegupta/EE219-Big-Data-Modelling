from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
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

#This function plots the ROC
def roc(test,classifier_soft,gam,i):
    probas_ = classifier_soft.predict_proba(train_tfidf_final[test])                                    
    fpr, tpr, thresholds = roc_curve(twenty_train.target[test], probas_[:, 1])
    s1 = 'SVM ROC for Gamma=%f' %  gam 
    s2 = ' & Fold=%d' % i
    s = s1+s2
    plt.plot(fpr, tpr, lw=1, label = s)                                    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc="lower right")
    plt.show()
    return

#This function varies the gamma everytime and generates a new SVM Model
def gamma_vary(gamma_curr):
    classifier_soft = svm.SVC(kernel='linear', probability=True,gamma = gamma_curr)
    print ("*****************************************************************************************")
    print ("The value of Gamma is ", gamma_curr)
    print ("*****************************************************************************************")

    for i, (train, test) in enumerate(cv):
        print ("\nFold Number: ", (i+1))
        Y_test_predicted = classifier_soft.fit(train_tfidf_final[train], train_target_final[train]).predict(train_tfidf_final[test])
        accuracy = np.mean(Y_test_predicted == twenty_train.target[test]) 
        print ("Current model Accuracy :-  ", accuracy)
        print ('\'0\' is Computer Technology and \'1\' is Recreational Activity')
        print ("Precision, recall values, f1-score,support are :-")      
        print (metrics.classification_report(train_target_final[test], Y_test_predicted))
        print ('Current confusion matrix :- ')
        print( metrics.confusion_matrix(train_target_final[test], Y_test_predicted))
        print ('Current ROC curve ')
        roc(test,classifier_soft,gamma_curr,i+1)
        print("########################################################################################")

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
twenty_train = fetch_20newsgroups(subset='all',categories=comp_rec, remove=('headers','footers','quotes'))

# Send all documents to get cleaned/processed for analysis
size, = twenty_train.filenames.shape
count =0
while count < size :
    temp = twenty_train.data[count]
    twenty_train.data[count] = clean_data(temp)
    count+=1


#Generating the DTM Matrix
count_vect = CountVectorizer()
twenty_td_vector = count_vect.fit_transform(twenty_train.data)

# Calculate the required txidf matrix by transorming the above formed vector and get categoriesteh required dimensions
tf_transformer = TfidfTransformer(use_idf=True).fit(twenty_td_vector)
twenty_tfidf_matrix = tf_transformer.transform(twenty_td_vector)
doc_count,term_count = twenty_tfidf_matrix.shape

#Dimensionalisty reduction to 50
#Applying LSI,SVD for normalization
#for training data
svd = TruncatedSVD(n_components=50, random_state=42)
train_tfidf_svd = svd.fit_transform(twenty_tfidf_matrix )
train_tfidf_final = Normalizer(copy=False).fit_transform(train_tfidf_svd)
train_target_final = twenty_train.target
for i in range(0,len(train_target_final)):
    if(train_target_final[i] <= 3):
        train_target_final[i] = 0
    else:
        train_target_final[i] = 1    


#Creating Folds
cv = StratifiedKFold(train_target_final, n_folds=5)

#Varying the values of Gamma
#gamma_arr = [0.001,0.01,0.1,1,10,100,1000]
gamma_arr = [0.01]
for i in range(0,len(gamma_arr)):
    gamma_vary(gamma_arr[i])