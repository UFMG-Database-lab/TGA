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