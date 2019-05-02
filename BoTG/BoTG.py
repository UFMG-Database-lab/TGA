"""
Created on Mon Nov 19 11:38:02 2018

@author: Vitor Mangaravite
"""
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions

from sklearn.metrics import pairwise_distances

from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SparkSession

from .DataRepresentation import Document

from scipy.sparse import lil_matrix, csr_matrix

import networkx as nx
import numpy as np

import psutil

import gc

from .Utils import _norm_graph_, _join_graph_, _get_subgraph_both_, _get_subgraph_in_, _get_subgraph_out_
from .dissimilatires import dissimilarity_node_in, dissimilarity_node_out, dissimilarity_node_both
from glob import glob
from tqdm import tqdm

from collections.abc import Iterable
import multiprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import cluster


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

        self._partition_size_ = 128

        self._model_rdd_=None

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
        return self
    def transform(self, X, pooling=None, assignment=None, format_doc=None, verbose=False):
        _format = self._validate_format_(format_doc)
        _assignment_ = self._get_assignment_function_(assignment)
        _pooling_ = self._get_pooling_function_(pooling)

        docs = self._get_documents_obj_(X, _format, verbose=verbose)
        
        X_result = self._tranform_pyspark_(docs, _assignment_, _pooling_)
        
        return X_result     

    # Assingment functions
    @staticmethod
    def _hard_idcf_assignment_(term, graph, dissimilarity_func, vector, clusters):
        if len(clusters) == 0:
            return vector
        values = [ (id_cluster, (1.-dissimilarity_func(graph, cluster_graph, term)), IDCF) for (id_cluster, cluster_graph, IDCF) in clusters ]
        
        idx_max, cluster_similarity, IDCF = max(values, key=lambda x: x[1] )
        vector[0,idx_max] =  graph.node[term]['TF'] * cluster_similarity * IDCF

        return vector
    @staticmethod
    def _hard_assignment_(term, graph, dissimilarity_func, vector, clusters):
        if len(clusters) == 0:
            return vector
        values = [ (id_cluster, (1.-dissimilarity_func(graph, cluster_graph, term))) for (id_cluster, cluster_graph, IDCF) in clusters ]

        idx_max = max(values, key=lambda x: x[1] )[0]
        vector[0,idx_max] = 1.
        return vector
    @staticmethod
    def _unorm_assignment_(term, graph, dissimilarity_func, vector, clusters):
        for (idx_cluster, cluster, IDCF) in clusters: #IDCF: Inverse Document-Context Frequency
            vector[0,idx_cluster] = (1.-dissimilarity_func(graph, cluster, term))
        #vector[0,idx_cluster] = vector[0,idx_cluster]/np.max(vector[0,idx_cluster])
        vector[0,] = vector[0,] / np.sum(vector[0,])
        return vector
    @staticmethod
    def _unorm_idcf_assignment_(term, graph, dissimilarity_func, vector, clusters):
        for (idx_cluster, cluster, IDCF) in clusters: #IDCF: Inverse Document-Context Frequency
            vector[0,idx_cluster] = graph.node[term]['TF'] * (1.-dissimilarity_func(graph, cluster, term)) * IDCF
        #vector[0,idx_cluster] = vector[0,idx_cluster]/np.max(vector[0,idx_cluster])
        vector[0,] = vector[0,] / np.sum(vector[0,])
        return vector
    def _get_assignment_function_(self, assignment):
        if assignment is None:
            assignment = self.assignment
        if callable(assignment):
            return assignment

        if assignment == 'hard':
            return BoTG._hard_assignment_
        if assignment == 'hard_idcf':
            return BoTG._hard_idcf_assignment_
            
        if assignment == 'unorm':
            return BoTG._unorm_assignment_
        if assignment == 'unorm_idcf':
            return BoTG._unorm_idcf_assignment_
        
        raise ValueError("%s assignment does not available." % assignment) 

    # Pooling functions
    @staticmethod
    def _mean_pooling_(X):
        return csr_matrix(X.mean(axis=0))
    @staticmethod
    def _max_pooling_(X):
        return csr_matrix(X.tocsr().max(axis=0))
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
    # ===================== Transform methods =====================
    def _tranform_pyspark_(self, list_of_docs, _assignment_, _pooling_):
        
        rdd_of_docs = self._sc_.parallelize(list_of_docs, self._partition_size_)

        rdd_of_docs = rdd_of_docs.zipWithIndex()
        rdd_of_docs = rdd_of_docs.flatMap( BoTG._build_each_term_vectors_(self._get_subgraph_) )

        joined_terms_doc = rdd_of_docs.join( self._model_rdd_ ) # result in [ (term, ((idx_doc, subgraph, DF), ([(idx_cluster,subcluster)]))) ] 
        joined_terms_doc = joined_terms_doc.map( BoTG._assignment_vector_(self.dissimilarity_func, _assignment_, self._n_clusters) )

        mapped_term_values = joined_terms_doc.groupByKey().mapValues(BoTG._convert_to_sparse_matrix_(self._n_clusters)).mapValues(_pooling_)
        #mapped_term_values.persist(StorageLevel.MEMORY_AND_DISK_SER)
        mapped_term_values.persist(StorageLevel.MEMORY_AND_DISK)
        
        result = mapped_term_values.collect()
        
        mapped_term_values.unpersist()
        del mapped_term_values

        new_result = lil_matrix( (len(list_of_docs), self._n_clusters), dtype=np.float64 )
        for idx,v in sorted(result, key=lambda x: x[0]):
            new_result[idx,] = v[0,]
        
        gc.collect()
        return new_result.tocsr()
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

    # ======================== Fit methods ========================
    def _build_clusters_pyspark_(self, list_of_docs, verbose=False):

        self._n_docs = len(list_of_docs)
        # (stage 0) Create repartitions
        rdd_of_docs = self._sc_.parallelize(list_of_docs, self._partition_size_)#.repartition(multiprocessing.cpu_count()*10)
        
        # (stage 1) flatMap: Create terms occurences
        index_of_docs = rdd_of_docs.flatMap( BoTG._create_index_pyspark_(self._get_subgraph_) )#.filter(lambda x: len(x[1].nodes) > 1)
        
        # (stage 2) groupByKey: Indexer
        index_of_docs = index_of_docs.groupByKey().mapValues(list)
        
        # Create matrix of co-occurrence, predict clusters and build term representations
        rdd_of_matrix = index_of_docs.map( BoTG._create_matrix_pyspark_(self.dissimilarity_func) )
        if self.metric != 'precomputed':
            rdd_of_matrix = rdd_of_matrix.map( BoTG._convert_matrix_pyspark_(self.metric) )
        rdd_of_matrix = rdd_of_matrix.map( BoTG._predict_clusterer_pyspark_(self.quantile) )
        rdd_of_matrix = rdd_of_matrix.flatMap( BoTG._build_term_representation_pyspark_ )
        rdd_of_matrix = rdd_of_matrix.filter( lambda x: len(x) > 0 )
        rdd_of_matrix = rdd_of_matrix.zipWithIndex()
        rdd_of_matrix = rdd_of_matrix.map( lambda x: (x[0][0], x[1], x[0][1]))
        rdd_of_matrix = rdd_of_matrix.map( BoTG._join_subgraphs_(self._n_docs) ).groupByKey().sortByKey().mapValues(list)

        self._model_rdd_ = rdd_of_matrix
        self._model_rdd_.persist(StorageLevel.MEMORY_AND_DISK)

        self._n_clusters = self._model_rdd_.map(lambda x: max( [xs[0] for xs in x[1]] ) ).max()+1
        self._unique_terms = self._model_rdd_.count()
        gc.collect()
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
                values = np.array([ diss_func(graph_i, graph_j, term) for graph_j in docs_within[j:] ])
                M[i,j:] = values
                M[j:,i] = values
            return (term, M, docs_within)
        return _co_create_matrix_pyspark_
    @staticmethod
    def _convert_matrix_pyspark_(metric):
        def _co_convert_matrix_pyspark_(x):
            term, M, docs_within = x
            return (term, pairwise_distances(M, metric=metric), docs_within)
        return _co_convert_matrix_pyspark_
    @staticmethod
    def _predict_clusterer_pyspark_(quantile):
        def _co_predict_clusterer_pyspark_(x):
            term, M, docs_within = x
            clusters = cluster.DBSCAN(n_jobs=1, eps=quantile, min_samples=int(np.log(M.shape[0])), metric="precomputed").fit_predict(M) 
            #clusters = cluster.DBSCAN(n_jobs=1, eps=quantile, min_samples=int(np.sqrt(M.shape[0])), metric="precomputed").fit_predict(M) 
            return (term, clusters, docs_within)
        return _co_predict_clusterer_pyspark_
    @staticmethod
    def _build_term_representation_pyspark_(x):
        (term, clusters, docs_within) = x
        mapper = [ (term, []) for i in range(max(clusters)+1) ]
        #if not len(mapper):
        #    return [ (term, [ docs_within[i] for (i,x) in enumerate(clusters) ]) ]
        list(map(lambda x: mapper[x[1]][1].append(docs_within[x[0]]), [ (i,x) for (i,x) in enumerate(clusters) if x >=0 ]))
        return mapper
    @staticmethod
    def _join_subgraphs_(N):
        def _co_join_subgraphs_(x):
            (term, idd, subgraphs) = x
            selected_subgraph = nx.DiGraph()
            for subgraph in subgraphs:
                selected_subgraph = _join_graph_(selected_subgraph, subgraph)
            selected_subgraph = _norm_graph_(selected_subgraph)
            IDCF = np.log( N / len(subgraphs) )
            return (term, (idd, selected_subgraph, IDCF ))
        return _co_join_subgraphs_

    # General Methods
    def _get_documents_obj_(self, X, format, verbose=False):
        if format == 'doc':
            return X
        if format == 'raw':
            return [ Document(text, lan=self.lang, w=self.w) for text in tqdm(X, smoothing=0., desc="Building documents", disable=not verbose) ]
        if format == 'filename':
            return Document.load_path(X, lan=self.lang, w=self.w)
    
    # spark methods
    def _get_default_spark_config_(self):
        cpu_count = multiprocessing.cpu_count()
        memory = psutil.virtual_memory().free
        
        executor_memory = max(int(0.8*memory), 5*1024*1024*1024)            # at least 5G
        driver_memory   = max((memory - executor_memory), 5*1024*1024*1024)

        spark_config = SparkConf()
        spark_config = spark_config.setAppName("BoTG_pySpark")
        spark_config = spark_config.setMaster("local[%d]" % cpu_count)
        
        spark_config = spark_config.set("spark.executor.memory", "%d" % executor_memory)
        spark_config = spark_config.set("spark.driver.memory", "%d" % driver_memory)

        spark_config = spark_config.set("spark.cleaner.periodicGC.interval", "1min")
        #spark_config = spark_config.set("spark.memory.fraction", "0.9")
        
        spark_config = spark_config.set("spark.executor.heartbeatInterval", "1000s")
        spark_config = spark_config.set("spark.network.timeout", "10000s")
        
        return spark_config
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
    
    # Validations
    def _set_direction_(self, direction):
        self.direction = direction
        if self.direction == 'out':
            self._get_subgraph_ = _get_subgraph_out_
            self.dissimilarity_func = dissimilarity_node_out
        elif self.direction == 'in':
            self._get_subgraph_ = _get_subgraph_in_
            self.dissimilarity_func = dissimilarity_node_in
        elif self.direction == 'both':
            self._get_subgraph_ = _get_subgraph_both_
            self.dissimilarity_func = dissimilarity_node_both
    def _validate_format_(self, format_doc):
        _format = format_doc
        if _format is None:
            _format = self.format
        if _format not in VALID_FORMATS:
            raise ValueError("%s format does not available." % _format)
        return _format
