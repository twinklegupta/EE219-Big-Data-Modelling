#importing the natural language processing and machine learning libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import nltk
import sys

Mas_File = "MATweets2.txt"
Wash_File = "WATweets2.txt"
Mas_Tweets = open(Mas_File, "r")
Wash_Tweets = open(Wash_File, "r")
#nltk.download()
#This function will perform the preprocessing of text
def clean(text):
    text = text.lower().decode('utf-8')
    tokenize1 = RegexpTokenizer(r'\w+')
    stemming = PorterStemmer()
    token = tokenize1.tokenize(text)
    result = [x for x in token if not x in stopwords.words('english')]
    result_new = [stemming.stem(y) for y in result]
    result_new = [z for z in result_new if not z.isdigit()]
    return " ".join(result_new)

MAlines = Mas_Tweets.read().split("$$DELIM$$")
WAlines = Wash_Tweets.read().split("$$DELIM$$")

#Splitting the data for training and testing 
X = len(MAlines)*25/100
Y = len(WAlines)*25/100

training = MAlines[X:] + WAlines[Y:]
training_target = [0]*(len(MAlines)-X) + [1]*(len(WAlines)-Y)

testing = MAlines[:X] + WAlines[:Y]
testing_target = [0]*X + [1]*Y

#We store the size of the dataset in the variable size_train and size_test
size_train = len(training)
size_test = len(testing)

# For every textual content of the tweet x, we pass it to the clean function 
# and overwrite the result
for x in range(size_train):
    text = training[x]
    training[x] = clean(text)

print "Reached this point"
for x in range(size_test):
    text = testing[x]
    testing[x] = clean(text)

#We  convert the resultant textual contents of tweet after preprocessing to a 
#term tweet matrix
cv = CountVectorizer()
X_train = cv.fit_transform(training)
X_test = cv.transform(testing)

# TF-IDF matrix for training
tfidf = TfidfTransformer(use_idf=True).fit(X_train)
resultant_matrix_train = tfidf.transform(X_train)
tweets_train,terms_train = resultant_matrix_train.shape

# TF-IDF matrix for testing
tfidf2 = TfidfTransformer(use_idf=True).fit(X_test)
resultant_matrix_test = tfidf2.transform(X_test)
tweets_test,terms_test = resultant_matrix_test.shape

#The truncatedSVD function gives us the lsi
lsi = TruncatedSVD(n_components=50, random_state=42)

#The lsi is then fitted and normalised
# training truncated SVD
X_train = lsi.fit_transform(resultant_matrix_train)
X_train = Normalizer(copy=False).fit_transform(X_train)
Y_train = training_target

# testing truncated SVD
X_test = lsi.transform(resultant_matrix_test)
X_test = Normalizer(copy=False).fit_transform(X_test)
Y_test = testing_target

######################################################
#Algorithm 1 - Support Vector Machine (SVM)
classifier = svm.SVC(kernel='linear', probability=True)
model = classifier.fit(X_train, Y_train)
Y_test_predicted = model.predict(X_test)

#Calculating accuracy, recall, precision and confusion matrix
accuracy_svm = np.mean(testing_target == Y_test_predicted)
print ('\'1\' is Washington and \'0\' is Massachusetts')
print ('The accuracy for the model is %f' % accuracy_svm)
print ("The precision and recall values are:")
print (metrics.classification_report(testing_target, Y_test_predicted))
print ('The confusion matrix is:')
print (metrics.confusion_matrix(testing_target, Y_test_predicted))

#Plotting the curve for ROC
probas_ = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(testing_target, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "Support Vector Machine - ROC")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("pl1.png")
plt.show()
#######################################################
#Algorithm 2 - Logistic Regression
logi_reg = LogisticRegression()
logi_reg.fit(X_train, Y_train)
Y_test_predicted = logi_reg.predict(X_test)

#Calculating accuracy, recall, precision and confusion matrix
accuracy_lr = np.mean(testing_target == Y_test_predicted)
print ('\'1\' is Washington and \'0\' is Massachusetts')
print ('The accuracy for the model is %f' % accuracy_lr)
print ("The precision and recall values are:")
print (metrics.classification_report(testing_target, Y_test_predicted))
print ('The confusion matrix is:')
print (metrics.confusion_matrix(testing_target, Y_test_predicted))

#Plotting the ROC
probas_ = logi_reg.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(testing_target, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "Logistic Regression - ROC")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("pl2.png")
plt.show()

#######################################################
#Algorithm 3 - Regularized Logistic Regression
logi_reg = LogisticRegression(penalty='l2', C=2.0, fit_intercept=True,
                              intercept_scaling=1, random_state=None, solver='liblinear',
                              multi_class='ovr', verbose=0, n_jobs=2)
logi_reg.fit(X_train, Y_train)
Y_test_predicted = logi_reg.predict(X_test)

#Calculating accuracy, recall, precision and confusion matrix
accuracy_lr = np.mean(testing_target == Y_test_predicted)
print ('\'1\' is Washington and \'0\' is Massachusetts')
print ('The accuracy for the model is %f' % accuracy_lr)
print ("The precision and recall values are:")
print (metrics.classification_report(testing_target, Y_test_predicted))
print ('The confusion matrix is:')
print (metrics.confusion_matrix(testing_target, Y_test_predicted))

#Plotting the ROC
probas_ = logi_reg.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(testing_target, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "Regularized Logistic Regression - ROC")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("pl3.png")
plt.show()
######################################################
#Algorithm 4 - Neural Network (NN)

##Testing for best hidden layer neurons - Answer = 100
#
#hidden = [50, 100, 500, 1000, 5000, 10000]
#max_accuracy = 0.0
#best_hidden_neurons = 50
#for i in hidden:  
#    mlp = MLPClassifier(hidden_layer_sizes=(i))
#    mlp.fit(X_train, Y_train)
#    Y_test_predicted = mlp.predict(X_test)
#    accuracy_lr = np.mean(testing_target == Y_test_predicted)
#    if accuracy_lr > max_accuracy:
#        max_accuracy = accuracy_lr
#        best_hidden_neurons = i

mlp = MLPClassifier(hidden_layer_sizes=(100))
mlp.fit(X_train, Y_train)
Y_test_predicted = mlp.predict(X_test)
accuracy_lr = np.mean(testing_target == Y_test_predicted)

#Calculating accuracy, recall, precision and confusion matrix
accuracy_nn = np.mean(testing_target == Y_test_predicted)
print ('\'1\' is Washington and \'0\' is Massachusetts')
print ('The accuracy for the model is %f' % accuracy_nn)
print ("The precision and recall values are:")
print (metrics.classification_report(testing_target, Y_test_predicted))
print ('The confusion matrix is:')
print (metrics.confusion_matrix(testing_target, Y_test_predicted))

#Plotting the ROC
probas_ = mlp.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(testing_target, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "Neural Network - ROC")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("pl4.png")
plt.show()

#######################################################
#Algorithm 5 - Naive Bayes (NB)
naive_model = GaussianNB()
naive_model.fit(X_train, Y_train)
Y_test_predicted = naive_model.predict(X_test)

#Calculating accuracy, recall, precision and confusion matrix
accuracy_nb = np.mean(testing_target == Y_test_predicted)
print ('\'1\' is Washington and \'0\' is Massachusetts')
print ('The accuracy for the model is %f' % accuracy_nb)
print ("The precision and recall values are:")
print (metrics.classification_report(testing_target, Y_test_predicted))
print ('The confusion matrix is:')
print (metrics.confusion_matrix(testing_target, Y_test_predicted))

#Plotting the ROC
probas_ = naive_model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(testing_target, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "Naive Bayes - ROC")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("pl5.png")
plt.show()



#'1' is Washington and '0' is Massachusetts
#The accuracy for the model is 0.764546
#The precision and recall values are:
#             precision    recall  f1-score   support
#
#          0       0.84      0.51      0.63      1905
#          1       0.74      0.94      0.83      2890
#
#avg / total       0.78      0.76      0.75      4795
#
#The confusion matrix is:
#[[ 963  942]
# [ 187 2703]]
#'1' is Washington and '0' is Massachusetts
#The accuracy for the model is 0.775600
#The precision and recall values are:
#             precision    recall  f1-score   support
#
#          0       0.83      0.55      0.66      1905
#          1       0.76      0.92      0.83      2890
#
#avg / total       0.78      0.78      0.76      4795
#
#The confusion matrix is:
#[[1049  856]
# [ 220 2670]]
#'1' is Washington and '0' is Massachusetts
#The accuracy for the model is 0.774557
#The precision and recall values are:
#             precision    recall  f1-score   support
#
#          0       0.82      0.55      0.66      1905
#          1       0.76      0.92      0.83      2890
#
#avg / total       0.78      0.77      0.76      4795
#
#The confusion matrix is:
#[[1050  855]
# [ 226 2664]]
#'1' is Washington and '0' is Massachusetts
#The accuracy for the model is 0.757039
#The precision and recall values are:
#             precision    recall  f1-score   support
#
#          0       0.73      0.62      0.67      1905
#          1       0.77      0.85      0.81      2890
#
#avg / total       0.75      0.76      0.75      4795
#
#The confusion matrix is:
#[[1172  733]
# [ 432 2458]]
#'1' is Washington and '0' is Massachusetts
#The accuracy for the model is 0.694056
#The precision and recall values are:
#             precision    recall  f1-score   support
#
#          0       0.60      0.68      0.64      1905
#          1       0.77      0.70      0.74      2890
#
#avg / total       0.70      0.69      0.70      4795
#
#The confusion matrix is:
#[[1292  613]
# [ 854 2036]]