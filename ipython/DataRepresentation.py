"""
Created on Mon Nov 19 11:38:02 2018

@author: Vitor Mangaravite
"""
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions

import networkx as nx
import numpy as np
import scipy.sparse as sp

from os import path
import math

from Utils import get_stopwords,get_stem,EXCLUDE
from glob import glob
from tqdm import tqdm

from collections.abc import Iterable

import multiprocessing
from multiprocessing import Pool

#from .MeanShift._meanshift_ import build_clusters


def K(x, sigma=100.):
    return np.exp( -(np.power(x,2.)/(2.*np.power(sigma,2.))) ) / (sigma*np.sqrt(2*np.pi))

class Document(object):
    def __init__(self, text, lan='en', w=1, kernel='norm'):
        self.text = text
        self.lan = lan
        self.w = w
        self.__build_graph__()
        self.__norm_graph__(kernel=self.__get_kernel__(kernel))
    @staticmethod
    def build_docs(docs_text, n_jobs=multiprocessing.cpu_count(), verbose=False, **kwargs):
        docs = []
        docs_text = [ (doc_text, kwargs) for doc_text in docs_text ]
        with Pool(processes=n_jobs) as p:
            for doc in tqdm(p.imap(Document._build_document_, docs_text), total=len(docs_text), desc="Building documents", disable=not verbose):
                docs.append(doc)
        return docs
    @staticmethod
    def build_sparse_matrices(docs_text, n_jobs=multiprocessing.cpu_count(), verbose=False, **kwargs):
        docs = []
        docs_text = [ (doc_text, kwargs) for doc_text in docs_text ]
        with Pool(processes=n_jobs) as p:
            for doc in tqdm(p.imap(Document._build_document_, docs_text), total=len(docs_text), desc="Building documents", disable=not verbose):
                yield (doc.sparse_linegraph() , doc.sparse_graph())
    @staticmethod
    def load_document(filepath, **kwargs):
        with open(filepath, "rb") as filin:
            text = filin.read().decode(errors='ignore')
        return Document(text, **kwargs)
    @staticmethod
    def _load_document_(param):
         return Document.load_document(param[0], **param[1])
    @staticmethod
    def _build_document_(param):
         return Document(param[0], **param[1])
    @staticmethod
    def load_path(filepattern, n_jobs=multiprocessing.cpu_count(), verbose=False, **kwargs):
        docs = []
        if isinstance(filepattern, str):
            docs_path = glob(filepattern)
        else:
            docs_path = filepattern
        docs_path = [ (filepath, kwargs) for filepath in docs_path ]
        with Pool(processes=n_jobs) as p:
            for doc in tqdm(p.imap(Document._load_document_, docs_path), total=len(docs_path), desc="Loading and building documents", disable=not verbose):
                docs.append(doc)
        return docs
    
    def __build_graph__(self):
        stopwords = get_stopwords(self.lan)
        stem = get_stem(self.lan).stem
        self.G = nx.Graph()
        sentences_str = [ [w for w in split_contractions(web_tokenizer(s)) if not (w.startswith("'") and len(w) > 1) and len(w) > 0] for s in list(split_multi(self.text)) if len(s.strip()) > 0]
        for sentence in sentences_str:
            buffer = []
            for word in sentence:
                if len([c for c in word if c in EXCLUDE]) == len(word) or word.lower() in stopwords or word.replace('.','').replace(',','').replace('-','').isnumeric():
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
        self.__build_linegraph__()
    def __build_linegraph__(self):
        self.LG = nx.Graph()
        for (s_source,t_source) in self.G.edges:
            (s_source,t_source) = (min(s_source,t_source), max(s_source,t_source))
            for (s_target, t_target) in self.G.edges([s_source, t_source]):
                (s_target, t_target) = (min(s_target, t_target),max(s_target, t_target))
                if (s_source,t_source) != (s_target, t_target):
                    self.LG.add_edge( (s_source,t_source), (s_target, t_target) )
    def sparse_linegraph(self, mapper=None):
        nodelist = list(self.LG.nodes)
        if len(nodelist) == 0:
            return sp.csr_matrix((0,0)), []
        return nx.to_scipy_sparse_matrix(self.LG, nodelist=nodelist), nodelist

    def sparse_graph(self, mapper=None):
        nodelist = list(self.G.nodes)
        if len(nodelist) == 0:
            return sp.csr_matrix((0,0)), []
        return nx.to_scipy_sparse_matrix(self.G, weight='weight', nodelist=nodelist), nodelist

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
