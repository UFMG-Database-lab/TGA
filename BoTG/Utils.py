"""
Created on Mon Nov 19 11:38:02 2018

@author: Vítor Mangaravite
"""
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import string
from os import path

lemma = WordNetLemmatizer()
stem  = PorterStemmer()

EXCLUDE = set(string.punctuation)
STOPWORDS = {}

def get_stopwords(lang):
    if lang not in STOPWORDS:
        STOPWORDS[lang] = load_stopword_list(lang)
    return STOPWORDS[lang]

def load_stopword_list(lang):
    with open(path.join(".", path.dirname(__file__), "StopwordsList", "stopwords_%s.txt" % lang[:2])) as stopword_file:
        stopwords = [ stop.replace('\n', '') for stop in stopword_file.readlines()]
    return set(stopwords) 

def get_stem(lan):
    return stem

def get_lemma(lan):
    return lemma