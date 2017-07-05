
"""
Created on Fri Jun 16 23:14:58 2017

@author: Sensei
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics


#url ="Real.csv" #naive 84%, KNN 57%, SVM 55%, Decision 68%
#url ="Testing.csv" #naive 84.48%, KNN 84.48%, SVM 82.75%, Decision 84.48%
#url ="trainCSV.csv" #naive 79%, KNN 66%, SVM 71%, Decision 76%
#url ="trainCleaned.csv" #naive 84%, KNN 57%, SVM 55%, Decision 68%
url ="data.csv" #naive 84.48%, KNN 84.48%, SVM 82.75%, Decision 84.48%



path = pd.read_csv(url , header = None, names=['insult','comments'])
#data = pd.read_csv(url)
print(path)
#print(path.head(10))
#print(path.insult.value_counts())
#print(path.comments.value_counts())

X = path.comments
y = path.insult
print(X.shape)
print(y.shape) 

X_train,X_test, y_train,y_test = train_test_split(X , y, random_state = 1)

#print(X_train.shape)
#print(X_test)
#print(y_train.shape)
#print(y_test)

vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
#X_train_dtm = vect.fit_transform(X_train)

X_test_dtm = vect.transform(X_test)
#print(X_test_dtm.shape)
#print(X_train_dtm)


nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
#print(Naive)

KNN = KNeighborsClassifier()
KNN.fit(X_train_dtm, y_train)

GNB = SVC()
GNB.fit(X_train_dtm, y_train)

DTC = DecisionTreeClassifier()
DTC.fit(X_train_dtm, y_train)

y_pred_classNB = nb.predict(X_test_dtm)
PredictNB = pd.DataFrame({'Actual':X_test, 'comments':y_test, 'predicted':y_pred_classNB})
#AccuracyNB = metrics.accuracy_score(y_test, PredictNB)
#print(PredictNB)
#print(accuracy_score(y_test, PredictNB))

y_pred_classKNN = KNN.predict(X_test_dtm)
PredictKNN = pd.DataFrame({'Actual':X_test, 'comments':y_test, 'predicted':y_pred_classKNN})
#AccuracyKNN = metrics.accuracy_score(y_test, PredictKNN)
#print(PredictKNN)
#print(AccuracyKNN)


y_pred_classGNB = GNB.predict(X_test_dtm)
PredictGNB = pd.DataFrame({'Actual':X_test, 'comments':y_test, 'predicted':y_pred_classKNN})
#AccuracyKNN = metrics.accuracy_score(y_test, PredictKNN)
#print(PredictGNB)
#print(AccuracyKNN)

y_pred_classGTC = DTC.predict(X_test_dtm)
PredictDTC = pd.DataFrame({'Actual':X_test, 'comments':y_test, 'predicted':y_pred_classNB})
#AccuracyNB = metrics.accuracy_score(y_test, PredictNB)
#print(PredictDTC)
#print(accuracy_score(y_test, PredictNB))

Accuracy1 = metrics.confusion_matrix(y_test, y_pred_classNB)
Accuracy = metrics.accuracy_score(y_test, y_pred_classNB)
print(Accuracy * 100)
print(Accuracy1)

Accuracy2 = metrics.confusion_matrix(y_test, y_pred_classKNN)
Accuracy3 = metrics.accuracy_score(y_test, y_pred_classKNN)
print(Accuracy3 * 100)
print(Accuracy2)


Accuracy4 = metrics.confusion_matrix(y_test, y_pred_classGNB)
Accuracy5 = metrics.accuracy_score(y_test, y_pred_classGNB)
print(Accuracy5 * 100)
print(Accuracy4)

Accuracy6 = metrics.confusion_matrix(y_test, y_pred_classGTC)
Accuracy7 = metrics.accuracy_score(y_test, y_pred_classGTC)
print(Accuracy7 * 100)
print(Accuracy6)


X_train_tokens = vect.get_feature_names()
insult_token_count = nb.feature_count_[0,:]
comments_token_count = nb.feature_count_[1,:]
hmm = nb.feature_count_
#print(hmm.shape)

#print(X_train_tokens[-50:])

tokens = pd.DataFrame({'token': X_train_tokens, 'insult':insult_token_count, 'comments':comments_token_count})
#print(tokens)

tokens['insult'] = tokens.insult + 1
tokens['comments'] = tokens.comments + 1

tokens['insult'] = tokens.insult/ nb.class_count_[0]
tokens['comments'] = tokens.comments/ nb.class_count_[1]

tokens['ratio'] = tokens.comments/ tokens.insult
hmm = tokens.sort_values('ratio',ascending= False)
print(hmm)