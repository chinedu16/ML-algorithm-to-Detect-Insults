# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:47:06 2017

@author: Sensei
"""

from nltk.corpus import stopwords
from nltk.tokenize import toktok

example_sentence = "this is an example of how stopwords can be filtered"
stop_word = set(stopwords.words("english"))

words = toktok(example_sentence)
filter_sentence = []

for w in words:
    if w not in stop_word:
        filter_sentence.append(w)
    
print(filter_sentence)