# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:46:11 2017

@author: Sensei
"""
# Load libraries
import pandas as pd

url ="Real.csv"
path = pd.read_table(url ,header = None, names=['insult','comments'])

print(path)