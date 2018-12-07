"""
Created on Mon Nov 19 11:38:02 2018

@author: Vitor Mangaravite
"""
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions

import networkx as nx
import numpy as np

from os import path
import math

from Utils import *
from glob import glob
from tqdm import tqdm

from collections.abc import Iterable
import multiprocessing
from MeanShift._meanshift_ import build_clusters
from sklearn.base import BaseEstimator, TransformerMixin

VALID_FORMATS = ['doc', 'raw', 'filename']

class BoTG(BaseEstimator, TransformerMixin): # based on TfidfTransformer structure
    def __init__(self, format_doc='doc', w=2, lang='en', min_df=2, n_jobs=None, max_iter=100, metric='cosine', quantile=0.001, pooling='mean', assignment='hard'):
        self.format = self._validate_format_(format_doc)
        self.w = 2
        self.lang = lang
        self.min_df = min_df
        self.max_iter = max_iter
        self.metric = metric
        self.quantile = quantile
        self.pooling = pooling
        self.assignment = assignment
        self.n_jobs = n_jobs if n_jobs is not None else multiprocessing.cpu_count()

        self._clusters = []
    def fit(self, X, y=None, format_doc=None, verbose=False, **fit_params):
        _format = self._validate_format_(format_doc)
        if 'pooling' in fit_params:
            self.pooling = fit_params['pooling']
        if 'assignment' in fit_params:
            self.assignment = fit_params['assignment']
        return self
    def transform(self, X, pooling=None, assignment=None, format_doc=None, verbose=False):
        _format = self._validate_format_(format_doc)
        pass
    

    # Validations
    def _validate_format_(self, format_doc):
        _format = format_doc
        if _format is None:
            _format = self.format
        if _format not in VALID_FORMATS:
            raise ValueError("%s format does not available." % _format)
        return _format


class Document(object):
    def __init__(self, text, id=None, lan='en', w=1, kernel='norm'):
        self.text = text
        self.id = id
        self.lan = lan
        self.w = w
        self.__build_graph__()
        self.__norm_graph__(kernel=self.__get_kernel__(kernel))
    @staticmethod
    def load_document(filepath, encoding='utf8', **kwargs):
        with open(filepath, "rb") as filin:
            text = filin.read().decode(errors='ignore')
        if 'id' not in kwargs:
            kwargs['id'] = path.basename(filepath)
        return Document(text, **kwargs)
    
    def __build_graph__(self):
        stopwords = get_stopwords(self.lan)
        stem = get_stem(self.lan).stem
        self.G = nx.DiGraph()
        sentences_str = [ [w for w in split_contractions(web_tokenizer(s)) if not (w.startswith("'") and len(w) > 1) and len(w) > 0] for s in list(split_multi(self.text)) if len(s.strip()) > 0]
        for sentence in sentences_str:
            buffer = []
            for word in sentence:
                if len([c for c in word if c in EXCLUDE]) == len(word): # If the word is based on exclude chars
                    buffer = []
                elif word.lower() in stopwords or word.replace('.','').replace(',','').replace('-','').isnumeric():
                    continue
                else:
                    #stemmed_word = lemma(word).lower()
                    stemmed_word = stem(word)
                    if stemmed_word not in self.G:
                        self.G.add_node(stemmed_word, TF=0)
                    self.G.node[stemmed_word]['TF'] += 1
                    for (idx_cooccur, word_cooccur) in enumerate(buffer[-self.w:]):
                        self.__add_cooccur__(word_cooccur, stemmed_word, idx_cooccur+1)
                    buffer.append(stemmed_word)
    def __norm_graph__(self, kernel): # const-kernel
        if not len(self.G.nodes):
            maxTF_vertex = 1.
        else:
            maxTF_vertex = max([ v_attr['TF'] for v, v_attr in self.G.nodes(data=True) ])
        for v, v_attr in self.G.nodes(data=True):
            self.G.node[v]['weight'] = v_attr['TF'] / maxTF_vertex
        
        if not len(self.G.edges):
            maxTF_edge = 1.
        else:
            maxTF_edge   = max([ kernel(e) for _,_,e in self.G.edges(data=True) ])
        for _,_,e_attr in self.G.edges(data=True):
            e_attr['weight'] = kernel(e_attr) / maxTF_edge
    def __add_cooccur__(self, left_term, right_term, idx_cooccur):
        if right_term not in self.G[left_term]:
            self.G.add_edge(left_term, right_term)
        if idx_cooccur not in self.G[left_term][right_term]:
            self.G[left_term][right_term][idx_cooccur] = 1.
        else:
            self.G[left_term][right_term][idx_cooccur] += 1.
    
    def __get_kernel__(self, kernel):
        if kernel == 'norm':
            return self.__kernel_const__
        elif kernel == 'log-norm' or kernel == 'log':
            return self.__kernel_log__
        else:
            raise ValueError('%s does not available as kernel' % kernel)
    def __kernel_const__(self, cooccurs):
        return sum(cooccurs.values())
    def __kernel_log__(self, cooccurs):
        return sum( [ value / math.log(pos+1) for pos, value in cooccurs.items() if isinstance(pos, int) ] )