# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:32:51 2017

@author: Sensei
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

train ="Testing.csv"
train_path = pd.read_csv(train, header = None, names=['insult','comments'])

test ="test.csv"
test_path = pd.read_csv(test, header = None, names=['Labelled','comment'])


#print(train_path["comments"])
#print(test_path)


#X = train_path.comments
#y = train_path.insult
#print(y.columns)
#print(y.shape) 
#test_Data = test_path

#X_train,X_test, y_train,y_test = train_test_split(X , y, random_state = 1)

#vect = CountVectorizer()

#vect.fit(X_train)
#X_train_dtm = vect.transform(X_train)
#X_train_dtm = vect.fit_transform(X_train)

vect = CountVectorizer()
vect.fit(train_path["comments"])
X_train_dtm = vect.transform(train_path["comments"])
X_train_dtm = vect.fit_transform(train_path["comments"])


vect.fit(test_path["comment"])
X_test_dtm = vect.transform(test_path["comment"])
X_test_dtm = vect.fit_transform(test_path["comment"])


nb = MultinomialNB()
Naive = nb.fit(X_train_dtm, train_path["insult"])

y_pred_classNB = nb.predict(X_test_dtm)
print(y_pred_classNB)
#PredictNB = pd.DataFrame({'Actual':test_path["labelled"], 'predicted':y_pred_classNB})

#y_pred_class = nb.predict(test_path)
#Accuracy1 = metrics.confusion_matrix(y_test, y_pred_class)
#Accuracy = metrics.accuracy_score(y_pred_class)
#print(Accuracy)
#print(Accuracy1)

#X_train_tokens = vect.get_feature_names()
#insult_token_count = nb.feature_count_[0,:]
#comments_token_count = nb.feature_count_[1,:]
#hmm = nb.feature_count_
#print(hmm.shape)
#print(X_train_tokens[-50:])

#tokens = pd.DataFrame({'token': X_train_tokens, 'insult':insult_token_count, 'comments':comments_token_count})

#tokens['insult'] = tokens.insult + 1
#tokens['comments'] = tokens.comments + 1

#tokens['insult'] = tokens.insult/ nb.class_count_[0]
#tokens['comments'] = tokens.comments/ nb.class_count_[1]

#tokens['ratio'] = tokens.comments/ tokens.insult
#hmm = tokens.sort_values('ratio',ascending= False)
#print(hmm)
