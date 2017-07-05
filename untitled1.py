# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 23:14:58 2017

@author: Sensei
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

url ="Real.csv"
path = pd.read_table(url ,header = None, names=['insult','comments'])


#data = pd.read_csv(url)
#print(path)
#print(path.head(10))
#print(path.insult.value_counts())
#print(path.comments.value_counts())

X = path.comments
y = path.insult
#print(X.shape)
#print(y.shape) 

X_train,X_test, y_train,y_test = train_test_split(X,y, random_state = 1)
#print(X_train.shape)
#print(X_test)
#print(y_train.shape)
#print(y_test)

vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_train_dtm = vect.fit_transform(X_train)

X_test_dtm = vect.transform(X_test)
#print(X_test_dtm.shape)
#print(X_train_dtm.shape)

nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
#print(Naive)

y_pred_class = nb.predict(X_test_dtm)
Prediction = pd.DataFrame({'Comment':X_test, 'Insult':y_test, 'predicted':y_pred_class})
print(Prediction)


Accuracy1 = metrics.confusion_matrix(y_test, y_pred_class)
Accuracy = metrics.accuracy_score(y_test, y_pred_class)
print(Accuracy * 100)
#print(Accuracy1)

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
#print(hmm)