"""
Created on Mon Nov 19 11:38:02 2018

@author: Vitor Mangaravite
"""
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions

from .DataRepresentation import Document

import networkx as nx
import numpy as np

from os import path
import math

from .Utils import *
from .dissimilatires import dissimilarity_node
from glob import glob
from tqdm import tqdm

from collections.abc import Iterable
import multiprocessing
from .MeanShift._meanshift_ import build_clusters
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
        self._bandwidth_terms = []
        self._bandwidth_clusters = []
    def fit(self, X, y=None, format_doc=None, verbose=False, **fit_params):
        _format = self._validate_format_(format_doc)
        if 'pooling' in fit_params:
            self.pooling = fit_params['pooling']
        if 'assignment' in fit_params:
            self.assignment = fit_params['assignment']
        
        docs = self._get_documents_obj(X, _format, verbose=verbose)
        terms_idx = self._build_term_idx_(docs, verbose=verbose)
        #self.statistics(docs, terms_idx)
        self._clusters = self._build_clusters_(docs, terms_idx, verbose=verbose)

        return self
    def transform(self, X, pooling=None, assignment=None, format_doc=None, verbose=False):
        _format = self._validate_format_(format_doc)
        pass
    def statistics(self, docs, terms_idx):
        terms_idx_ = [ (term, docs_within) for (term, docs_within) in terms_idx.items() if len(docs_within) >= self.min_df ]
        sizes = []
        for term, docs_within in tqdm(terms_idx_, desc="Analysing terms idx size", position=0):
            sizes.append(len(docs_within))
        bins_count, bins = np.histogram(sizes, bins=10)
        for i in range(len(bins)-1):
            print(bins[i], bins[i+1], bins_count[i])
    
    # private methods
    def _build_clusters_(self, docs, terms_idx, verbose=False):
        clusters = []
        terms_idx_ = [ (term, docs_within) for (term, docs_within) in terms_idx.items() if len(docs_within) >= self.min_df ]
        terms_idx_ = sorted(terms_idx_, key=lambda x: len(x[1]))
        for term, docs_within in tqdm(terms_idx_, desc="Building clusters", position=0, disable=not verbose):
            docs_within = list(docs_within)
            M = np.zeros((len(docs_within),len(docs_within)))
            for i, doc_i in enumerate(docs_within):
                for j, doc_j in enumerate(docs_within[:i]):
                    M[i,j] = M[i,j] = 1.-dissimilarity_node(doc_i.G, doc_j.G, term)
            result = build_clusters(M, n_jobs=self.n_jobs, max_iter=self.max_iter, quantile=self.quantile, metric=self.metric, verbose=verbose)
            

    def _get_documents_obj(self, X, format, verbose=False):
        if format == 'doc':
            return X
        if format == 'raw':
            return [ Document(text, lan=self.lang, w=self.w) for text in tqdm(X, desc="Building documents", disable=not verbose) ]
        if format == 'filename':
            return [ Document.load_document(filename, lan=self.lang, w=self.w) for filename in tqdm(X, desc="Loading and building documents", disable=not verbose) ]
    def _build_term_idx_(self, docs, verbose=False):
        terms_idx = {}
        for doc in tqdm(docs, desc="Building term index", disable=not verbose):
            for v_term in doc.G.nodes:
                if v_term not in terms_idx:
                    terms_idx[v_term] = []
                terms_idx[v_term].append( doc )
        return terms_idx
    # Validations
    def _validate_format_(self, format_doc):
        _format = format_doc
        if _format is None:
            _format = self.format
        if _format not in VALID_FORMATS:
            raise ValueError("%s format does not available." % _format)
        return _format
