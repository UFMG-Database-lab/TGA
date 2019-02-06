"""
Created on Mon Nov 19 11:38:02 2018

@author: Vitor Mangaravite
"""
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions

from .DataRepresentation import Document

import networkx as nx
import numpy as np
import pandas as pd

from os import path
import math

import psutil, sys, operator

import gc

from .Utils import *
from .dissimilatires import dissimilarity_node_in, dissimilarity_node_out, dissimilarity_node_both, dissimilarity_row
from glob import glob
from tqdm import tqdm

from collections.abc import Iterable
import multiprocessing
from multiprocessing import Pool, Process
#from .MeanShift._meanshift_ import build_clusters
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors 
from sklearn import cluster

import random

from joblib import Parallel, delayed
import time

VALID_FORMATS = ['doc', 'raw', 'filename']

def garbage_collector():
    while True:
        gc.collect()
        time.sleep(5)

def K(x, sigma=100.):
    return np.exp( -(np.power(x,2.)/(2.*np.power(sigma,2.))) ) / (sigma*np.sqrt(2*np.pi))        

def size_item(item, size_float):
    term, docs_within = item
    #return len(docs_within)*len(docs_within)*size_float
    return 8*(len(docs_within)*len(docs_within)*size_float + 2*sys.getsizeof(docs_within))

def process_term(params):
    term, docs_within, quantile, metric, dissimilarity_func, verbose = params
    docs_within = list(docs_within)
    M = np.eye(len(docs_within), dtype=np.float)
    qtd_total = int((len(docs_within)*len(docs_within))/2 - len(docs_within))
    with tqdm(total=qtd_total, position=2, desc="Building Distances", disable=not verbose, smoothing=0.) as pbar:
        for i, doc_i in enumerate(docs_within):
            j = i+1
            M[i,j:] = M[j:,i] = [ 1.-dissimilarity_func(doc_i.G, doc_j.G, term) for doc_j in docs_within[j:] ]
            pbar.update(len(docs_within)-j)
    #M = NearestNeighbors(metric=metric).fit(M).radius_neighbors_graph(mode='distance')
    eps = 0.1
    #if len(M.nonzero()[0]) > 0:
    #    eps = np.percentile(M[M.nonzero()].ravel(), q=quantile)
    min_samples = int(np.sqrt(M.shape[0]))
    dbscan = cluster.DBSCAN(n_jobs=1, eps=eps, min_samples=min_samples, metric=metric)
    clusters = dbscan.fit_predict(M)
    
    return term, clusters

class BoTG(BaseEstimator, TransformerMixin): # based on TfidfTransformer structure
    def __init__(self, format_doc='doc', w=2, lang='en', min_df=2,
    n_jobs=None, max_iter=100, direction='both', metric='cosine', memory_strategy='soft', quantile=0.01, pooling='mean', assignment='hard'):
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

        self.direction = direction
        if self.direction == 'out':
            self.dissimilarity_func = dissimilarity_node_out
        elif self.direction == 'in':
            self.dissimilarity_func = dissimilarity_node_in
        elif self.direction == 'both':
            self.dissimilarity_func = dissimilarity_node_both
        
        self.memory_strategy = memory_strategy
        if self.memory_strategy == 'norm':
            self._chunk_strategy = self._define_chunks_
        elif self.memory_strategy == 'hard':
            self._chunk_strategy = self._define_chunks_hard_
        elif self.memory_strategy == 'soft':
            self._chunk_strategy = self._define_chunks_soft_

    def fit(self, X, y=None, format_doc=None, verbose=False, **fit_params):
        _format = self._validate_format_(format_doc)
        if 'pooling' in fit_params:
            self.pooling = fit_params['pooling']
        if 'assignment' in fit_params:
            self.assignment = fit_params['assignment']
        
        docs = self._get_documents_obj_(X, _format, verbose=verbose)
        terms_idx = self._build_term_idx_(docs, verbose=verbose)
        if "save_dist" in fit_params and fit_params['save_dist']:
            self._save_distances_(docs, terms_idx, '../terms_matrix/', verbose=verbose)
        else:
            self._new_build_clusters_(docs, terms_idx, verbose=verbose) # 6/79734 [17:00]
            #self._build_clusters_(docs, terms_idx, verbose=verbose)      # 1/79734 [22:31]

        return self
    def transform(self, X, pooling=None, assignment=None, format_doc=None, verbose=False):
        _format = self._validate_format_(format_doc)
        _assignment_ = self._get_assignment_function_(assignment)
        _pooling_ = self._get_pooling_function_(pooling)
        docs = self._get_documents_obj_(X, _format, verbose=verbose)
        X_result = []
        for doc in tqdm(docs, desc="Building representation", total=len(docs), position=0, disable=not verbose, smoothing=0.):
            terms_assignments = []
            for term in tqdm(doc.G.nodes, desc="Building terms", total=len(doc.G.nodes), position=1, disable=not verbose, smoothing=0.):
                terms_assignments.append( _assignment_(term, doc.G) )
            X_result.append( _pooling_(np.array(terms_assignments)) )
        return np.array(X_result)
    
    # private methods

    # Assignment functions
    #self._clusters = [ subgraph ]
    #self._labels = [ (term, id_cluster) ]
    #self._labels_map = { term: [id_cluster] }
    def _hard_assignment_(self, term, graph):
        result = np.zeros(len(self._clusters))
        if term not in self._labels_map:
            return result
        values = [ (id_cluster, self.dissimilarity_func(graph, self._clusters[id_cluster], term)) for id_cluster in self._labels_map[term] ]
        j = min(values, key=lambda x: x[1] )[0]
        result[j] = 1.
        return result
    def _soft_assignment_(self, term, graph):
        result = np.ones(len(self._clusters))
        if term not in self._labels_map:
            return result / result.sum()
        j, values = list(zip(*[ (id_cluster, self.dissimilarity_func(graph, self._clusters[id_cluster], term)) for id_cluster in self._labels_map[term] ]))
        result[list(j)] = list(values)
        result = K(result)
        return result / result.sum()
    def _unorm_assignment_(self, term, graph):
        result = np.zeros(len(self._clusters))
        if term not in self._labels_map:
            return result
        j, values = list(zip(*[ (id_cluster, 1.-self.dissimilarity_func(graph, self._clusters[id_cluster], term)) for id_cluster in self._labels_map[term] ]))
        result[list(j)] = list(values)
        return result / result.sum()
    def _get_assignment_function_(self, assignment):
        if assignment is None:
            assignment = self.assignment
        if callable(assignment):
            return assignment
        if assignment == 'hard':
            return self._hard_assignment_
        if assignment == 'soft':
            return self._soft_assignment_
        if assignment == 'unorm':
            return self._unorm_assignment_
        raise ValueError("%s assignment does not available." % assignment) 

    # Pooling functions
    def _mean_pooling_(self, X):
        return X.mean(axis=0)
    def _max_pooling_(self, X):
        return X.max(axis=0)
    def _sum_pooling_(self, X):
        return X.sum(axis=0)
    def _get_pooling_function_(self, pooling):
        if pooling is None:
            pooling = self.pooling
        if callable(pooling):
            return pooling
        if pooling == 'mean':
            return self._mean_pooling_
        if pooling == 'max':
            return self._max_pooling_
        if pooling == 'sum':
            return self._sum_pooling_
        raise ValueError("%s pooling does not available." % pooling) 
    
    # Algorithm methods
    def _statistics_(self, terms_idx):
        terms_idx_ = [ (term, docs_within) for (term, docs_within) in terms_idx if len(docs_within) >= self.min_df ]
        sizes = []
        for _, docs_within in terms_idx_:
            sizes.append(len(docs_within))
        n_bins = min( 10, max(1, int(len(sizes)/2)) )
        bins_count, bins = np.histogram(sizes, bins=n_bins)
        for i in range(len(bins)-1):
            if bins_count[i] > 0:
                print("   [%d;%d[ = %d" % (round(bins[i],0), round(bins[i+1],0), bins_count[i]))
    def _save_distances_(self, docs, terms_idx, path_to_save, verbose=True):
        terms_idx_ = [ (term, docs_within) for (term, docs_within) in terms_idx.items() if len(docs_within) >= self.min_df ]
        terms_idx_ = sorted(terms_idx_, key=lambda x: len(x[1]), reverse=True)
        for term, docs_within in tqdm(terms_idx_, desc="Building clusters", position=0, disable=not verbose):
            docs_within = list(docs_within)
            M = np.eye(len(docs_within), dtype=np.float)
            total_iters = int((len(docs_within)*len(docs_within)-len(docs_within))/2)
            with tqdm(total=total_iters, desc="Building distances", position=1, disable=not verbose) as pbar:
                for i, doc_i in enumerate(docs_within):
                    for j, doc_j in enumerate(docs_within[:i]):
                        M[i,j] = M[j,i] = 1.-self.dissimilarity_func(doc_i.G, doc_j.G, term)
                        pbar.update(1)
            np.savetxt("%s/%s.csv" % (path_to_save, term), M, delimiter=",")
    def _new_build_clusters_(self, docs, terms_idx, verbose=False):
        self._clusters = []
        self._labels = []
        self._labels_map = {}
        garbage_process = Process(target=garbage_collector)
        garbage_process.start()

        terms_idx_ = [ (term, list(docs_within)) for (term, docs_within) in terms_idx.items() if len(docs_within) >= self.min_df ]
        terms_idx_ = sorted(terms_idx_, key=lambda x: len(x[1]), reverse=True)
        
        chunks = list(self._chunk_strategy(terms_idx_, verbose=verbose))
        if verbose:
            print("Chunked process:")
            for i, terms_idx_chunk in enumerate(chunks):
                end_chars = 's\n'
                if len(terms_idx_chunk) == 1:
                    end_chars = ''
                print(" iter=%d with %d term" % (i, len(terms_idx_chunk)), end=end_chars)
                self._statistics_(terms_idx_chunk)

        for terms_idx_chunk in tqdm(chunks, total=len(chunks), position=0, desc="Running chunks", disable=not verbose, smoothing=0.):
            params = self._make_params_(terms_idx_chunk, verbose)
            with Pool(processes=self.n_jobs) as p:
                for j,(term, cluster) in enumerate(tqdm(p.imap_unordered(process_term, params), smoothing=0., total=len(terms_idx_chunk), position=1, desc="Building Clusters", disable=not verbose)):
                    self._labels_map[term] = []
                    docs_within = terms_idx[term]
                    mapper = [ [] for i in range(max(cluster)+1) ]
                    list(map(lambda x: mapper[x[1]].append(docs_within[x[0]].G), [ (i,x) for (i,x) in enumerate(cluster) if x >=0 ]))
                    for i, doc_in_cluster in enumerate(mapper):
                        # map subgraphs
                        subgraphs = list(map(lambda x: self._get_subgraph_(x, term) , doc_in_cluster))
                        # reduce subgraphs
                        selected_subgraph = [] # subgraph
                        self._labels.append( (term, (i, len(self._clusters))) )
                        self._labels_map[term].append(len(self._clusters))
                        self._clusters.append( selected_subgraph )
                    """
                    result = build_clusters(M, n_jobs=self.n_jobs, max_iter=self.max_iter, quantile=self.quantile, metric=self.metric, verbose=verbose)
                    self._labels_map[term] = []
                    for i, id_cluster in enumerate(result['clusters'].keys()):
                        selected_subgraph = self._get_subgraph_(docs_within[id_cluster].G, term)
                        self._labels.append( (term, (i, len(self._clusters))) )
                        self._labels_map[term].append(len(self._clusters))
                        self._clusters.append( selected_subgraph )
                        # Se for construir uma representação sintética do cluster, usar
                        # o conjunto dos ids armazenados em result['clusters'][id_cluster]
                        # para recuperar os respectivos documentos em docs_within
                    """
        garbage_process.terminate()
                    #(_, cluster) = process_term_parallel( (term, docs, self.n_jobs, self.quantile, self.metric, dissimilarity_func, verbose) )

        #clusters = Parallel(n_jobs=self.n_jobs)(delayed(process_term)(term, docs, self.quantile, self.n_jobs, self.metric, dissimilarity_func) for (term, docs) in tqdm(terms_idx_, desc="Building clusters", position=1, disable=not verbose))
    def _define_chunks_hard_(self, array_to_chunk, verbose=False):
        aval = psutil.virtual_memory().available
        aval -= 0.5*aval

        #size_float = np.float64(0).nbytes
        size_float = sys.getsizeof(np.float64(0))
        
        finished_chunks = []
        chunks = [ (size_item(array_to_chunk[0], size_float), [array_to_chunk[0]]) ]
        
        for item in tqdm(array_to_chunk[1:], total=len(array_to_chunk)-1, desc="Defining chunks", disable=not verbose, smoothing=0.):
            create_new_chunk = True
            size_atual_item = size_item(item, size_float)
            chunks = sorted(chunks, key=lambda x: x[0])
            for id_chunk in range(len(chunks)):
                size_chunk, chunk = chunks[id_chunk]
                if (size_chunk + size_atual_item) < aval:
                    chunk.append(item)
                    if len(chunk) == self.n_jobs:
                        finished_chunks.append( chunk )
                        chunks.pop(id_chunk)
                    else:
                        chunks[id_chunk] = ((size_chunk+size_atual_item), chunk)
                    create_new_chunk = False
                    break
            if create_new_chunk:
                if len(finished_chunks) > 0:
                    id_chunk = min(enumerate([ len(chunk) for chunk in finished_chunks ]), key=operator.itemgetter(1))[0]
                    finished_chunks[id_chunk].append(item)
                else:
                    chunks.append( (size_atual_item,[item]) )
        finished_chunks.extend( [ chunk for _,chunk in chunks ] )
        #finished_chunks = list(reversed(finished_chunks))
        list(map(random.shuffle,finished_chunks))
        return finished_chunks
    def _define_chunks_(self, array_to_chunk, verbose=False):
        array_to_chunk_2 = array_to_chunk
        random.shuffle(array_to_chunk_2)
        return [ array_to_chunk_2 ]
    def _define_chunks_soft_(self, array_to_chunk, verbose=False):
        bins_count, _ = np.histogram([ len(item[1])^2 for item in array_to_chunk ], bins=10)
        size_bins = int(sum(bins_count[1:]))
        chunks = [[] for _ in range(size_bins)]
        rev_ = True
        for i, item in enumerate(array_to_chunk):
            idx = i % len(chunks)
            if idx == 0:
                rev_ = not rev_
            if rev_:
                idx = len(chunks)-idx-1
            chunks[idx].append(item)
        #chunks = list(reversed(chunks))
        #list(map(random.shuffle,chunks))
        return chunks
    def _define_chunks_soft_2_(self, array_to_chunk, verbose=False):
        i = 0
        j = len(array_to_chunk)-1
        
        bins_count, _ = np.histogram([ len(item[1])^2 for item in array_to_chunk ], bins=10)
        size_bins = sum(bins_count[1:])

        chunks_aux = []
        while i < j:
            big = array_to_chunk[i]
            size_big = len(big[1])^2
            chunk = [big]
            size_small = 0
            while ((size_small+len(array_to_chunk[j][1])^2) <= size_big or len(chunk) < size_bins) and i != j:
                small = array_to_chunk[j]
                chunk.append(small)
                size_small += len(small[1])^2
                j -= 1
            chunks_aux.append(chunk)
            i += 1
        chunks = [[] for _ in range(size_bins)]
        rev_ = True
        for i in range(len(chunks_aux)):
            idx = i % len(chunks)
            if idx == 0:
                rev_ = not rev_
            if rev_:
                idx = len(chunks)-idx-1
            chunks[idx].extend(chunks_aux[i])
        #chunks = list(reversed(chunks))
        #list(map(random.shuffle,chunks))
        return chunks
    def _get_subgraph_(self, G, term):
        g = nx.DiGraph()
        g.add_edges_from( G.edges(term, data=True) )
        if len(g.nodes) == 0:
            # this term-node does not have any edge
            g.add_node(term)
        for n in g.nodes:
            for att in G.node[n]:
                g.node[n][att] = G.node[n][att]
        return g
    def _get_documents_obj_(self, X, format, verbose=False):
        if format == 'doc':
            return X
        if format == 'raw':
            return [ Document(text, lan=self.lang, w=self.w) for text in tqdm(X, smoothing=0., desc="Building documents", disable=not verbose) ]
        if format == 'filename':
            return Document.load_path(X, lan=self.lang, w=self.w)
    def _build_term_idx_(self, docs, verbose=False):
        terms_idx = {}
        for doc in tqdm(docs, desc="Building term index", disable=not verbose):
            for v_term in doc.G.nodes:
                if v_term not in terms_idx:
                    terms_idx[v_term] = []
                terms_idx[v_term].append( doc )
        return terms_idx
    def _make_params_(self,terms_idx_chunk, verbose):
        for i, (term, docs) in enumerate(terms_idx_chunk):
            yield (term, docs, self.quantile, self.metric, self.dissimilarity_func, i == 0 and verbose)
    # Validations
    def _validate_format_(self, format_doc):
        _format = format_doc
        if _format is None:
            _format = self.format
        if _format not in VALID_FORMATS:
            raise ValueError("%s format does not available." % _format)
        return _format

#docs = Document.load_path('sample/*', verbose=True, w=3)
#botg = BoTG()
#botg.fit(docs, verbose=True)
#botg.transform(docs, assignment='soft', verbose=True)

# import importlib
# importlib.reload(BoTG.BoTG)
