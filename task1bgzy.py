import csv
import pandas as pd
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np

#using some natural language processing code(with nltk) given in the workshop4

commonword_m = ['software', 'Software', 'Inc.', 'corporation', 'Corporation', 'development', 'inc.', 'system', 'System', 'mac', 'Mac','inc','Inc','.','encore']

amaz = pd.read_csv("amazon.csv")
goog = pd.read_csv("google.csv")
len_g = len(goog)
len_a = len(amaz)
amaz_assign = {}
goog_assign = {}
amaz_m = {}
goog_m = {}
for i in range(len_a):
    speech = amaz['title'][i];
    wordList = nltk.word_tokenize(speech)
    stopWords = set(stopwords.words('english'))
    filteredList = [w for w in wordList if not w in stopWords and  not w in commonword_m];
    amaz_assign[amaz['idAmazon'][i]] = filteredList


for i in range(len_a):
    speech = amaz['manufacturer'][i];
    if speech:
        wordList = nltk.word_tokenize(speech)
        stopWords = set(stopwords.words('english'))
        filteredList = [w for w in wordList if not w in stopWords and  not w in commonword_m];
        amaz_m[amaz['idAmazon'][i]] = filteredList


for i in range(len_g):
    speech = goog['name'][i];
    wordList = nltk.word_tokenize(speech)
    stopWords = set(stopwords.words('english'))
    filteredList = [w for w in wordList if not w in stopWords and not w in commonword_m ];
    goog_assign[goog['id'][i]] = filteredList



for i in range(len_g):
    speech = goog['manufacturer'][i];
    if type(speech) != float:
        wordList = nltk.word_tokenize(speech)
        stopWords = set(stopwords.words('english'))
        filteredList = [w for w in wordList if not w in stopWords and  not w in commonword_m];
        goog_m[goog['id'][i]] = filteredList




with open('google_blocks.csv', 'a', newline = '') as f:
    addtool = csv.writer(f)
    addtool.writerow(['block_key', 'product_id'])
    for key in goog_assign.keys():
        for block_key in goog_assign[key]:
            if block_key.isalpha():
                addtool.writerow([block_key, key])
    for key in goog_m.keys():
        for block_key in goog_m[key]:
            addtool.writerow([block_key, key])
f.close()


with open('amazon_blocks.csv', 'a', newline = '') as f:
    addtool = csv.writer(f)
    addtool.writerow(['block_key', 'product_id'])
    for key in amaz_assign.keys():
        for block_key in amaz_assign[key]:
            if block_key.isalpha():
                addtool.writerow([block_key, key])
    for key in amaz_m.keys():
        for block_key in amaz_m[key]:
            addtool.writerow([block_key, key])
f.close()
