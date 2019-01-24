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

from .Utils import *
from .dissimilatires import dissimilarity_node_in, dissimilarity_node_out, dissimilarity_node_both, dissimilarity_row
from glob import glob
from tqdm import tqdm

from collections.abc import Iterable
import multiprocessing
from multiprocessing import Pool
#from .MeanShift._meanshift_ import build_clusters
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn import cluster

from joblib import Parallel, delayed
import time

VALID_FORMATS = ['doc', 'raw', 'filename']

def K(x, sigma=100.):
    return np.exp( -(np.power(x,2.)/(2.*np.power(sigma,2.))) ) / (sigma*np.sqrt(2*np.pi))        

def size_item(item, size_float):
    term, docs_within = item
    return 2.5*(len(docs_within)*len(docs_within)*size_float + sys.getsizeof(docs_within))    

def process_term(params):
    term, docs_within, quantile, metric, dissimilarity_func = params
    docs_within = list(docs_within)
    M = np.eye(len(docs_within), dtype=np.float)
    for i, doc_i in enumerate(docs_within):
        for j, doc_j in enumerate(docs_within[:i]):
            M[i,j] = M[j,i] = 1.-dissimilarity_func(doc_i.G, doc_j.G, term)
    eps = np.percentile(M.ravel(), q=quantile)
    min_samples = int(np.sqrt(M.shape[0]))
    dbscan = cluster.DBSCAN(n_jobs=1, eps=eps, min_samples=min_samples, metric=metric)
    clusters = dbscan.fit_predict(M)
    return term, clusters

class BoTG(BaseEstimator, TransformerMixin): # based on TfidfTransformer structure
    def __init__(self, format_doc='doc', w=2, lang='en', min_df=2,
    n_jobs=None, max_iter=100, direction='both', metric='cosine', quantile=0.01, pooling='mean', assignment='hard'):
        self.format = self._validate_format_(format_doc)
        self.w = 2
        self.lang = lang
        self.min_df = min_df
        self.max_iter = max_iter
        self.metric = metric
        self.quantile = quantile
        self.pooling = pooling
        self.direction = direction
        self.assignment = assignment
        self.n_jobs = n_jobs if n_jobs is not None else multiprocessing.cpu_count()

        
    def fit(self, X, y=None, format_doc=None, verbose=False, **fit_params):
        _format = self._validate_format_(format_doc)
        if 'pooling' in fit_params:
            self.pooling = fit_params['pooling']
        if 'assignment' in fit_params:
            self.assignment = fit_params['assignment']
        
        docs = self._get_documents_obj_(X, _format, verbose=verbose)
        terms_idx = self._build_term_idx_(docs, verbose=verbose)
        if verbose:
                self._statistics_(docs, terms_idx)
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
        for doc in tqdm(docs, desc="Building representation", total=len(docs), position=0, disable=not verbose):
            terms_assignments = []
            for term in tqdm(doc.G.nodes, desc="Building terms", total=len(doc.G.nodes), position=1, disable=not verbose):
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
        values = [ (id_cluster, dissimilarity_node(graph, self._clusters[id_cluster], term)) for id_cluster in self._labels_map[term] ]
        j = min(values, key=lambda x: x[1] )[0]
        result[j] = 1.
        return result
    def _soft_assignment_(self, term, graph):
        result = np.ones(len(self._clusters))
        if term not in self._labels_map:
            return result / result.sum()
        j, values = list(zip(*[ (id_cluster, dissimilarity_node(graph, self._clusters[id_cluster], term)) for id_cluster in self._labels_map[term] ]))
        result[list(j)] = list(values)
        result = K(result)
        return result / result.sum()
    def _unorm_assignment_(self, term, graph):
        result = np.zeros(len(self._clusters))
        if term not in self._labels_map:
            return result
        j, values = list(zip(*[ (id_cluster, 1.-dissimilarity_node(graph, self._clusters[id_cluster], term)) for id_cluster in self._labels_map[term] ]))
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
    def _statistics_(self, docs, terms_idx):
        terms_idx_ = [ (term, docs_within) for (term, docs_within) in terms_idx.items() if len(docs_within) >= self.min_df ]
        sizes = []
        for _, docs_within in tqdm(terms_idx_, desc="Analysing terms idx size", position=0):
            sizes.append(len(docs_within))
        bins_count, bins = np.histogram(sizes, bins=10)
        print("Statistics of terms idx sizes:")
        for i in range(len(bins)-1):
            print(" [%d;%d[ = %d" % (round(bins[i],0), round(bins[i+1],0), bins_count[i]))
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
                        M[i,j] = M[j,i] = 1.-dissimilarity_node_both(doc_i.G, doc_j.G, term)
                        pbar.update(1)
            np.savetxt("%s/%s.csv" % (path_to_save, term), M, delimiter=",")
    def _new_build_clusters_(self, docs, terms_idx, verbose=False):
        self._clusters = []
        self._labels = []
        self._labels_map = {}

        if self.direction == 'out':
            dissimilarity_func = dissimilarity_node_out
        elif self.direction == 'in':
            dissimilarity_func = dissimilarity_node_in
        elif self.direction == 'both':
            dissimilarity_func = dissimilarity_node_both

        terms_idx_ = [ (term, docs_within) for (term, docs_within) in terms_idx.items() if len(docs_within) >= self.min_df ]
        terms_idx_ = sorted(terms_idx_, key=lambda x: len(x[1]), reverse=True)
        
        chunks = list(reversed(self._define_chunks_2_(terms_idx_, verbose=verbose)))
        if verbose:
            print("Chunked process:")
            for i, terms_idx_chunk in enumerate(chunks):
                print(" iter=%d with %d terms" % (i, len(terms_idx_chunk)))

        for terms_idx_chunk in tqdm(chunks, total=len(chunks), position=0, desc="Running chunks", disable=not verbose):
            params = [(term, docs, self.quantile, self.metric, dissimilarity_func) for (term, docs) in terms_idx_chunk]
            with Pool(processes=self.n_jobs) as p:
                for term, cluster in tqdm(p.imap(process_term, params), total=len(params), position=1, desc="Building Clusters", disable=not verbose):
                    pass

        #clusters = Parallel(n_jobs=self.n_jobs)(delayed(process_term)(term, docs, self.quantile, self.n_jobs, self.metric, dissimilarity_func) for (term, docs) in tqdm(terms_idx_, desc="Building clusters", position=1, disable=not verbose))
    def _define_chunks_2_(self, array_to_chunk, verbose=False):
        aval = psutil.virtual_memory().available
        
        size_float = np.float64(0).nbytes
        finished_chunks = []
        chunks = [ (size_item(array_to_chunk[0], size_float), [array_to_chunk[0]]) ]
        
        for item in tqdm(array_to_chunk[1:], total=len(array_to_chunk)-1, desc="Defining chunks", disable=not verbose):
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
        return finished_chunks
    def _define_chunks_(self, array_to_chunk):
        import psutil, sys

        aval = psutil.virtual_memory().available
        qtd_threads = self.n_jobs
        max_size_chunk = 1./qtd_threads * aval
        
        #size_float = sys.getsizeof(np.float64(0))
        size_float = np.float64(0).nbytes
        
        chunks = []
        chunks_atual = []
        for (term, docs_within) in array_to_chunk:
            n_items = len(list(docs_within))
            #size_chunk = n_items*n_items*size_float # uma matriz quadratica de floats

            # uma matriz quadratica de floats + tamanho da lista dos documentos (considerando a copia dos objs e um overhead de +50% [hard overhead])
            size_chunk = 2.5*(n_items*n_items*size_float + sys.getsizeof(docs_within))
            
            old_qtd_thread = qtd_threads
            while size_chunk > max_size_chunk and qtd_threads > 1:
                qtd_threads -= 1
                max_size_chunk = 1./qtd_threads * aval
            if qtd_threads != old_qtd_thread:
                chunks.append( (old_qtd_thread, chunks_atual) )
                chunks_atual = []
            chunks_atual.append( (term, docs_within) )
        chunks.append( (old_qtd_thread, chunks_atual) )
        return chunks

    def _build_matrix_(self, docs_within, term, verbose=False):
        term_graph_list = [ nx.to_pandas_edgelist(self._get_subgraph_(doc.G, term)) for doc in tqdm(docs_within, desc="Building edgelist", position=1, disable=not verbose) ]
        term_weight_list = []
        id_doc_to_avalable = []
        df = pd.DataFrame()
        for i, df_subgraph in tqdm(enumerate(term_graph_list), desc="Building term-matrix", position=1, total=len(term_graph_list), disable=not verbose):
            if df_subgraph.size > 0:
                df_subgraph['__docid__'] = i
                df_subgraph.set_index(['source', '__docid__'])
                df = df.append(df_subgraph, sort=True)
                term_weight_list.append(docs_within[i].G.node[term]['weight'])
                id_doc_to_avalable.append(i)
        if df.size < 2:
            return None, None
        df = df.pivot_table(index=['source', '__docid__'], columns='target', values='weight')
        matrix = df.values
        term_weight_list = np.matrix(term_weight_list).T
        matrix = np.nan_to_num(np.array(np.c_[ term_weight_list, matrix ]))
        print(' \n\n Size matrix', matrix.shape, end='')
        M = pairwise_distances(matrix, metric=dissimilarity_row, n_jobs=self.n_jobs)
        print(M.shape)

        return M, id_doc_to_avalable
    def _build_clusters_(self, docs, terms_idx, verbose=False):
        self._clusters = []
        self._labels = []
        self._labels_map = {}

        if self.direction == 'out':
            dissimilarity_func = dissimilarity_node_out
        elif self.direction == 'in':
            dissimilarity_func = dissimilarity_node_in
        elif self.direction == 'both':
            dissimilarity_func = dissimilarity_node_both

        terms_idx_ = [ (term, docs_within) for (term, docs_within) in terms_idx.items() if len(docs_within) >= self.min_df ]
        terms_idx_ = sorted(terms_idx_, key=lambda x: len(x[1]), reverse=True)

        total_itens = sum( [ int(((len(docs_within)*len(docs_within))-len(docs_within))/2) for _, docs_within in terms_idx_ ] )
        with tqdm(desc="Building distances", total=total_itens, position=0, disable=not verbose) as pbar:
            for term, docs_within in tqdm(terms_idx_, desc="Building clusters", position=1, disable=not verbose):
                docs_within = list(docs_within)
                M = np.eye(len(docs_within), dtype=np.float64)

                for i, doc_i in enumerate(docs_within):
                    #all_res = Parallel(n_jobs=self.n_jobs)(delayed(dissimilarity_node_both) (doc_i.G, doc_j.G, term) for doc_j in docs_within[:i])
                    #M[i,:i] = M[:i,i] = 1.-np.array(all_res)
                    for j, doc_j in enumerate(docs_within[:i]):
                        M[i,j] = M[j,i] = 1.-dissimilarity_func(doc_i.G, doc_j.G, term)
                        pbar.update(1)
            eps = np.percentile(M.ravel(), q=self.quantile)
            min_samples = int(np.sqrt(M.shape[0]))
            dbscan = cluster.DBSCAN(n_jobs=self.n_jobs, eps=eps, min_samples=min_samples, metric=self.metric)
            clusters = dbscan.fit_predict(M)
            # result = { 'bandwidth': bandwidth, 'clusters':clusters, 'mapper_cluster': mapper_subgraph_cluster }
            """result = build_clusters(M, n_jobs=self.n_jobs, max_iter=self.max_iter, quantile=self.quantile, metric=self.metric, verbose=verbose)
            self._labels_map[term] = []
            for i, id_cluster in enumerate(result['clusters'].keys()):
                selected_subgraph = self._get_subgraph_(docs_within[id_cluster].G, term)
                self._labels.append( (term, (i, len(self._clusters))) )
                self._labels_map[term].append(len(self._clusters))
                self._clusters.append( selected_subgraph )
                # Se for construir uma representação sintética do cluster, usar
                # o conjunto dos ids armazenados em result['clusters'][id_cluster]
                # para recuperar os respectivos documentos em docs_within"""
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
            return [ Document(text, lan=self.lang, w=self.w) for text in tqdm(X, desc="Building documents", disable=not verbose) ]
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
