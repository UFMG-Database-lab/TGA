"""
Created on Mon Nov 19 11:38:02 2018

@author: Vitor Mangaravite
"""
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions

from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SparkSession

from .DataRepresentation import Document

import concurrent.futures

#from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool as ThreadPool

from scipy.sparse import lil_matrix, csr_matrix

import networkx as nx
import numpy as np
import pandas as pd

from os import path
import math

import psutil, sys, operator

import gc

from .Utils import *
from .Utils import process_term, garbage_collector, _norm_graph_, _join_graph_, _get_subgraph_both_, _get_subgraph_in_, _get_subgraph_out_
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

from contextlib import closing

import random

from joblib import Parallel, delayed

def process_chunk(params):
    chunk, params_func, verbose = params
    with Pool(processes=1) as executor:
        for (term, clusters) in tqdm(executor.imap_unordered(process_term, params_func(chunk, verbose)), smoothing=0., total=len(chunk), position=1, desc="Building Clusters", disable=not verbose):
            yield (term, clusters)

VALID_FORMATS = ['doc', 'raw', 'filename']

class BoTG(BaseEstimator, TransformerMixin): # based on TfidfTransformer structure
    def __init__(self, format_doc='doc', w=2, lang='en', min_df=2,
    direction='both', metric='cosine', quantile=0.1, pooling='mean', assignment='hard', spark=None):
        self.format = self._validate_format_(format_doc)
        self.w = 2
        self.lang = lang
        self.min_df = min_df
        self.metric = metric
        self.quantile = quantile
        self.pooling = pooling
        self.assignment = assignment

        self._set_spark_(spark)
        self._set_direction_(direction)

        self._model_rdd_=None
        """
        self.memory_strategy = memory_strategy
        if self.memory_strategy == 'norm':
            self._chunk_strategy = self._define_chunks_
        elif self.memory_strategy == 'hard':
            self._chunk_strategy = self._define_chunks_hard_
        elif self.memory_strategy == 'soft':
            self._chunk_strategy = self._define_chunks_soft_
        """
    def __str__(self):
        if self._model_rdd_ is None:
            return "<BoTG(assig=%s, pooling=%s, metric=%s, direction=%s)>" % (self.assignment, self.pooling, self.metric, self.direction)
        return "<BoTG(assig=%s, pooling=%s, metric=%s, direction=%s, nclusters=%d, unique_terms=%d)>" % (self.assignment, self.pooling, self.metric, self.direction, self._n_clusters, self._unique_terms)

    def fit(self, X, y=None, format_doc=None, verbose=False, **fit_params):
        _format = self._validate_format_(format_doc)
        if 'pooling' in fit_params:
            self.pooling = fit_params['pooling']
        if 'assignment' in fit_params:
            self.assignment = fit_params['assignment']
        
        docs = self._get_documents_obj_(X, _format, verbose=verbose)

        self._build_clusters_pyspark_(docs, verbose=verbose)

        self._fit_=True

        #terms_idx = self._build_term_idx_(docs, verbose=verbose)
        #self._build_clusters_(docs, terms_idx, verbose=verbose)

        return self
    # Return a csr sparse matrix
    def transform(self, X, pooling=None, assignment=None, format_doc=None, verbose=False):
        _format = self._validate_format_(format_doc)
        _assignment_ = self._get_assignment_function_(assignment)
        _pooling_ = self._get_pooling_function_(pooling)
        docs = self._get_documents_obj_(X, _format, verbose=verbose)

        X_result = self._tranform_pyspark_(docs, _assignment_, _pooling_)
        #X_result = np.array([ _pooling_(terms_assignments) for terms_assignments in [ [ _assignment_(term, doc.G) for term in doc.G.nodes ] for doc in tqdm(docs, desc="Building representation", total=len(docs), position=0, disable=not verbose, smoothing=0.) ] ])        
        #for doc in tqdm(docs, desc="Building representation", total=len(docs), position=0, disable=not verbose, smoothing=0.):
        #    terms_assignments = np.array([ _assignment_(term, doc.G) for term in doc.G.nodes ])
        #    X_result.append( _pooling_(terms_assignments) )
        return X_result
        
    def _tranform_pyspark_(self, list_of_docs, _assignment_, _pooling_):
        
        rdd_of_docs = self._sc_.parallelize(list_of_docs, 100)

        rdd_of_docs = rdd_of_docs.zipWithIndex().flatMap( BoTG._build_each_term_vectors_(self._get_subgraph_) )#.filter( lambda x: len(x[1][1].nodes) > 1 )

        joined_terms_doc = rdd_of_docs.join( self._model_rdd_ ) # result in [ (term, ((idx_doc, subgraph), ([(idx_cluster,subcluster)]))) ] 
        joined_terms_doc = joined_terms_doc.map( BoTG._assignment_vector_(self.dissimilarity_func, _assignment_, self._n_clusters) )

        #rdd_of_docs = rdd_of_docs
        
        mapped_term_values = joined_terms_doc.groupByKey().mapValues(BoTG._convert_to_sparse_matrix_(self._n_clusters)).mapValues(_pooling_)

        mapped_term_values.persist(StorageLevel.MEMORY_AND_DISK_SER)
        
        result = mapped_term_values.collect()

        mapped_term_values.unpersist()
        del mapped_term_values
        gc.collect()

        new_result = lil_matrix( (len(list_of_docs), self._n_clusters), dtype=np.float64 )
        for idx,v in sorted(result, key=lambda x: x[0]):
            new_result[idx,] = v[0,]
        
        return new_result.tocsr()
        
        """
        X_result = lil_matrix( (len(list_of_docs),len(self._clusters)), dtype=np.float64 )
        for (i,doc) in tqdm(enumerate(list_of_docs), desc="Building representation", total=len(list_of_docs), position=0, smoothing=0.):
            terms_assignments = lil_matrix( (len(doc.G.nodes),len(self._clusters)), dtype=np.float64 )
            for (j, term) in enumerate(doc.G.nodes):
                terms_assignments[j,] = _assignment_(term, doc.G)
            X_result[i,] = _pooling_( terms_assignments )
            del terms_assignments
        """
        
    @staticmethod
    def _build_each_term_vectors_(_get_subgraph_):
        def _co_build_each_term_vectors_(x):
            doc, idx = x
            return [ (node, (idx, _get_subgraph_(doc.G, node))) for node in doc.G.nodes ]
        return _co_build_each_term_vectors_
    
    @staticmethod
    def _assignment_vector_(dissimilarity_func, _assignment_func_, n_cluster):
        def _co_assignment_vector_(x):
            (term, ((idx_doc, graph), clusters )) = x
            vector = _assignment_func_(term, graph, dissimilarity_func, lil_matrix((1,n_cluster), dtype=np.float64), clusters)
            return (idx_doc, vector)
        return _co_assignment_vector_
    @staticmethod
    def _convert_to_sparse_matrix_(n_clusters):
        def _co_convert_to_sparse_matrix_(X):
            X = list(X)
            matrix = lil_matrix((len(X),n_clusters), dtype=np.float64)
            for i,x in enumerate(X):
                matrix[i,] = x
            return matrix
        return _co_convert_to_sparse_matrix_

    # private methods

    # Assignment functions
    #self._clusters = [ subgraph ]
    #self._labels = [ (term, id_cluster) ]
    #self._labels_map = { term: [id_cluster] }
    

    @staticmethod
    def _hard_assignment_(term, graph, dissimilarity_func, vector, clusters):
        if len(clusters) == 0:
            return vector
        values = [ (id_cluster, 1.-dissimilarity_func(graph, cluster_graph, term)) for (id_cluster, cluster_graph) in clusters ]

        idx_max = max(values, key=lambda x: x[1] )[0]
        vector[0,idx_max] = 1.
        
        """
        maximum = max(values, key=lambda x: x[1] )[1]
        indices = [i for (i, v) in enumerate(values) if v == maximum]
        for idx in indices:
            result[0,idx] = 1.
        """
        return vector
    @staticmethod
    def _unorm_assignment_(term, graph, dissimilarity_func, vector, clusters):
        # result = np.zeros(len(self._clusters))
        for (idx_cluster, cluster) in clusters:
            vector[0,idx_cluster] = 1.-dissimilarity_func(graph, cluster, term)
        return vector / vector.sum()
    def _get_assignment_function_(self, assignment):
        if assignment is None:
            assignment = self.assignment
        if callable(assignment):
            return assignment
        if assignment == 'hard':
            return BoTG._hard_assignment_
        if assignment == 'soft':
            return BoTG._soft_assignment_
        if assignment == 'unorm':
            return BoTG._unorm_assignment_
        raise ValueError("%s assignment does not available." % assignment) 

    # Pooling functions
    @staticmethod
    def _mean_pooling_(X):
        return csr_matrix(X.mean(axis=0))
    @staticmethod
    def _max_pooling_(X):
        return csr_matrix(X.max(axis=0))
    @staticmethod
    def _sum_pooling_(X):
        return csr_matrix(X.sum(axis=0))
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
    # PySpark methods
    def _build_clusters_pyspark_(self, list_of_docs, verbose=False):
        #sc.setLogLevel('INFO' if verbose else 'ERROR')
        
        # (stage 0) Create repartitions
        rdd_of_docs = self._sc_.parallelize(list_of_docs, 100)#.repartition(multiprocessing.cpu_count()*10)
        
        # (stage 1) flatMap: Create terms occurences
        index_of_docs = rdd_of_docs.flatMap( BoTG._create_index_pyspark_(self._get_subgraph_) )#.filter(lambda x: len(x[1].nodes) > 1)
        
        # (stage 2) groupByKey: Indexer
        index_of_docs = index_of_docs.groupByKey().mapValues(list)
        
        # Create matrix of co-occurrence, predict clusters and build term representations
        rdd_of_matrix = index_of_docs.map( BoTG._create_matrix_pyspark_(self.dissimilarity_func) )
        rdd_of_matrix = rdd_of_matrix.map( BoTG._predict_clusterer_pyspark_(self.quantile, self.metric) )
        rdd_of_matrix = rdd_of_matrix.flatMap( BoTG._build_term_representation_pyspark_ )
        rdd_of_matrix = rdd_of_matrix.filter( lambda x: len(x[1]) > 0 )
        rdd_of_matrix = rdd_of_matrix.zipWithIndex().map( lambda x: (x[0][0], x[1], x[0][1]))
        rdd_of_matrix = rdd_of_matrix.map( BoTG._join_subgraphs_ ).groupByKey().sortByKey().mapValues(list)

        self._model_rdd_ = rdd_of_matrix
        self._model_rdd_.persist(StorageLevel.MEMORY_AND_DISK_SER)

        self._n_clusters = self._model_rdd_.map(lambda x: max( [xs[0] for xs in x[1]] ) ).max()+1
        self._unique_terms = self._model_rdd_.count()

        """
        # (stage 3) collect: get the ResultStage
        
        result_clusters = rdd_of_matrix.collect()

        rdd_of_matrix.unpersist()
        del rdd_of_matrix
        
        self._n_clusters = 0
        for (term, clusters) in result_clusters:
            self._labels_map[term] = []
            for selected_subgraph in clusters:
                self._labels_map[term].append( (self._n_clusters, selected_subgraph) )
                self._n_clusters += 1
        self._n_clusters 

        if self._sc_ is None:
            sc.stop()
            del sc
        """
        gc.collect()
    # try https://stackoverflow.com/questions/32505426/how-to-process-rdds-using-a-python-class
    @staticmethod
    def _create_index_pyspark_(_get_subgraph_):
        def _co_create_index_pyspark_(doc):
            return [(node, _get_subgraph_(doc.G, node)) for node in doc.G.nodes]
        return _co_create_index_pyspark_
    @staticmethod
    def _create_matrix_pyspark_(diss_func):
        def _co_create_matrix_pyspark_(x):
            (term, docs_within) = x
            M = np.zeros((len(docs_within), len(docs_within)), dtype=np.float)
            for i, graph_i in enumerate(docs_within):
                j = i+1
                values = np.array([ 1.-diss_func(graph_i, graph_j, term) for graph_j in docs_within[j:] ])
                M[i,j:] = values
                M[j:,i] = values
            return (term, M, docs_within)
        return _co_create_matrix_pyspark_
    @staticmethod
    def _predict_clusterer_pyspark_(quantile, metric):
        def _co_predict_clusterer_pyspark_(x):
            term, M, docs_within = x
            clusters = cluster.DBSCAN(n_jobs=1, eps=quantile, min_samples=int(np.sqrt(M.shape[0])), metric=metric).fit_predict(M) 
            return (term, clusters, docs_within)
        return _co_predict_clusterer_pyspark_
    @staticmethod
    def _build_term_representation_pyspark_(x):
        (term, clusters, docs_within) = x

        (term, clusters, docs_within) = x
        mapper = [ (term, []) for i in range(max(clusters)+1) ]
        list(map(lambda x: mapper[x[1]][1].append(docs_within[x[0]]), [ (i,x) for (i,x) in enumerate(clusters) if x >=0 ]))
        return mapper
        # return (term, mapper) 

        # then flatMap(_build_term_representation_pyspark_).mapValues( "_join_subgraphs_" ).mapValues( _norm_graph_ ).reduceByKey( "concat" )
        """
        _clusters = []
        mapper = [ [] for i in range(max(clusters)+1) ]
        list(map(lambda x: mapper[x[1]].append(docs_within[x[0]]), [ (i,x) for (i,x) in enumerate(clusters) if x >=0 ]))
        for subgraphs in mapper:
            # reduce subgraphs
            selected_subgraph = nx.DiGraph()
            for subgraph in subgraphs:
                selected_subgraph = _join_graph_(selected_subgraph, subgraph)
            selected_subgraph = _norm_graph_(selected_subgraph)
            _clusters.append( selected_subgraph )
        return (term, _clusters)
        """
    @staticmethod
    def _join_subgraphs_(x):
        (term, idd, subgraphs) = x
        selected_subgraph = nx.DiGraph()
        for subgraph in subgraphs:
            selected_subgraph = _join_graph_(selected_subgraph, subgraph)
        selected_subgraph = _norm_graph_(selected_subgraph)
        return (term, (idd, selected_subgraph))

    # General Methods
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
    def _build_clusters_(self, docs, terms_idx, verbose=False):
        self._clusters = []
        self._labels = []
        self._labels_map = {}
        garbage_process = Process(target=garbage_collector)
        garbage_process.start()

        terms_idx_ = [ (term, list(docs_within)) for (term, docs_within) in terms_idx.items() if len(docs_within) >= self.min_df ]
        terms_idx_ = sorted(terms_idx_, key=lambda x: len(x[1]), reverse=True)

        #with ThreadPool(processes=self.n_jobs) as executor:
        sc = SparkContext('local', 'pyspark tutorial') 
        df = pd.DataFrame(list(self._make_params_(terms_idx_, verbose)))
        params = sc.parallelize(df)

        print(params)

        for (term, clusters) in params.map(process_term).collect():
            if len(clusters) > 0:
                self._labels_map[term] = []
                for i, selected_subgraph in enumerate(clusters):
                    self._labels.append( (term, (i, len(self._clusters))) )
                    self._labels_map[term].append(len(self._clusters))
                    self._clusters.append( selected_subgraph )
        #with Pool(processes=self.n_jobs) as executor:
        #    for (term, clusters) in tqdm(executor.imap_unordered(process_term, params), total=len(terms_idx_), position=0, desc="Running chunks", disable=not verbose, smoothing=0.):     
        """
        chunks = list(self._chunk_strategy(terms_idx_, verbose=verbose))
        if verbose:
            print("Chunked process:")
            for i, terms_idx_chunk in enumerate(chunks):
                end_chars = 's\n'
                if len(terms_idx_chunk) == 1:
                    end_chars = ''
                print(" iter=%d with %d term" % (i, len(terms_idx_chunk)), end=end_chars)
                self._statistics_(terms_idx_chunk)
                
        #with ThreadPool(processes=self.n_jobs) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            #params = self._make_params_(terms_idx_chunk, verbose)
            for to_unpack in tqdm(executor.map(process_chunk, [ (chunk, self._make_params_, verbose) for chunk in chunks ]), total=len(chunks), position=0, desc="Running chunks", disable=not verbose, smoothing=0.):
                for (term, clusters) in to_unpack:
                    if not len(clusters):
                        self._labels_map[term] = []
                        for i, selected_subgraph in enumerate(clusters):
                            self._labels.append( (term, (i, len(self._clusters))) )
                            self._labels_map[term].append(len(self._clusters))
                            self._clusters.append( selected_subgraph )
        """
        garbage_process.terminate()
        #pool = Pool(processes=self.n_jobs)
        #with closing(Pool(processes=self.n_jobs)) as executor:
        #with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
        #with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            #for terms_idx_chunk in tqdm(chunks, total=len(chunks), position=0, desc="Running chunks", disable=not verbose, smoothing=0.):
                #params = self._make_params2_(terms_idx_chunk, pool, verbose)
                
                
                #for (term, clusters) in tqdm(executor.map(process_term_func, params), smoothing=0., total=len(terms_idx_chunk), position=1, desc="Building Clusters", disable=not verbose):
                #for (term, clusters) in tqdm(executor.imap_unordered(process_term_func, params), smoothing=0., total=len(terms_idx_chunk), position=1, desc="Building Clusters", disable=not verbose):
                #future_to_terms = [executor.submit(process_term_func, param) for param in params]
                #for future in tqdm(concurrent.futures.as_completed(future_to_terms), smoothing=0., total=len(terms_idx_chunk), position=1, desc="Building Clusters", disable=not verbose):
                #    (term, clusters) = future.result()
                #for (term, cluster) in tqdm(Parallel(n_jobs=self.n_jobs)(process_term(param) for param in params), smoothing=0., total=len(terms_idx_chunk), position=1, desc="Building Clusters", disable=not verbose):
                #for (term, cluster) in tqdm(Parallel(n_jobs=self.n_jobs)(process_term_func(param) for param in params), smoothing=0., total=len(terms_idx_chunk), position=1, desc="Building Clusters", disable=not verbose):
                #for param in tqdm(params, smoothing=0., total=len(terms_idx_chunk), position=1, desc="Building Clusters", disable=not verbose):
                #    (term, clusters) = process_term(param)
                    #if not len(clusters):
                        #self._labels_map[term] = []
                        #for i, selected_subgraph in enumerate(clusters):
                            #self._labels.append( (term, (i, len(self._clusters))) )
                            #self._labels_map[term].append(len(self._clusters))
                            #self._clusters.append( selected_subgraph )
                #gc.collect()
                #gc.collect(1)
                #gc.collect(2)
    def _get_default_spark_config_(self):
        cpu_count = multiprocessing.cpu_count()
        memory = psutil.virtual_memory().free
        #memory = psutil.virtual_memory().available
        executor_memory = max(int(0.3*memory), 471859200)
        driver_memory = memory - executor_memory
        spark_config = SparkConf()
        spark_config = spark_config.setAppName("BoTG_pySpark")
        spark_config = spark_config.setMaster("local[%d]" % cpu_count)
        
        spark_config = spark_config.set("spark.executor.memory", "%d" % executor_memory)
        #spark_config = spark_config.set("spark.executor.memory", "%d" % int(executor_memory*0.5))
        #spark_config = spark_config.set("spark.executor.pyspark.memory", "%d" % int(executor_memory*0.5))

        spark_config = spark_config.set("spark.driver.memory", "%d" % driver_memory)

        spark_config = spark_config.set("spark.cleaner.periodicGC.interval", "1min")

        spark_config = spark_config.set("spark.memory.fraction", "0.9")
        
        spark_config = spark_config.set("spark.executor.heartbeatInterval", "1000s")
        spark_config = spark_config.set("spark.network.timeout", "10000s")
        
        return spark_config
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
        finished_chunks = list(reversed(finished_chunks))
        list(map(random.shuffle,finished_chunks))
        return finished_chunks
    def _define_chunks_(self, array_to_chunk, verbose=False):
        #random.shuffle(array_to_chunk)
        array_to_chunk_2 = list(reversed(array_to_chunk))
        return [ array_to_chunk_2 ]
    def _define_chunks_soft_(self, array_to_chunk, verbose=False):
        bins_count, _ = np.histogram([ len(item[1])^2 for item in array_to_chunk ], bins=10)
        size_bins = int(sum(bins_count[9:]))
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
    def _size_neighborhood_both_(self, term, G):
        return len(G.edges(term)) + len(G.in_edges(term))
    def _size_neighborhood_in_(self, term, G):
        return len(G.in_edges(term))
    def _size_neighborhood_out_(self, term, G):
        return len(G.edges(term))
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
                size_nei = self._size_neighborhood_(v_term, doc.G)
                if not size_nei:
                    continue
                if v_term not in terms_idx:
                    terms_idx[v_term] = []
                terms_idx[v_term].append( doc )
        return terms_idx
    def _make_params_(self,terms_idx_chunk, verbose):
        for i, (term, docs) in enumerate(terms_idx_chunk):
            yield (term, docs, self.quantile, self.metric, self.dissimilarity_func, self._get_subgraph_, verbose)
    def _make_params2_(self,terms_idx_chunk, pool, verbose):
        for i, (term, docs) in enumerate(terms_idx_chunk):
            yield (term, docs, self.quantile, self.metric, self.dissimilarity_func, self._get_subgraph_, pool, i == 0 and verbose)
    def _set_spark_(self, spark):
        if spark is None:
            self.spark_config = self._get_default_spark_config_()
            self._sc_ = SparkContext(conf=self.spark_config).getOrCreate()
        elif isinstance(spark, SparkConf):
            self.spark_config = spark
            self._sc_ = SparkContext(conf=self.spark_config).getOrCreate()
        elif isinstance(spark, SparkContext):
            self._sc_ = spark
            self.spark_config = self._sc_.getConf()
        elif isinstance(spark, SparkSession):
            self._sc_ = spark.sparkContext
            self.spark_config = self._sc_.getConf()
    def close(self):
        self._sc_.stop()
    def _set_direction_(self, direction):
        self.direction = direction
        if self.direction == 'out':
            self._get_subgraph_ = _get_subgraph_out_
            self._size_neighborhood_ = self._size_neighborhood_out_
            self.dissimilarity_func = dissimilarity_node_out
        elif self.direction == 'in':
            self._get_subgraph_ = _get_subgraph_in_
            self._size_neighborhood_ = self._size_neighborhood_in_
            self.dissimilarity_func = dissimilarity_node_in
        elif self.direction == 'both':
            self._get_subgraph_ = _get_subgraph_both_
            self._size_neighborhood_ = self._size_neighborhood_both_
            self.dissimilarity_func = dissimilarity_node_both
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
