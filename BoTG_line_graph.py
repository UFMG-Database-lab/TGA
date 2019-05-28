import networkx as nx
import numpy as np
from tqdm import tqdm
from BoTG.DataRepresentation import Document
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import pool
import random
from scipy.sparse import lil_matrix

def metric(neighbors1, neighbors2):
    set_neigh1 = set(neighbors1)
    set_neigh2 = set(neighbors2)
    t_intersect = len(set_neigh1.intersection(set_neigh2))
    return t_intersect/len(set_neigh1)

def co_app2(x):
    (((source, neigh_source), (target, neigh_target)),weight) = x
    return ((source, target), weight*compare_jac(neigh_source,neigh_target))
def compare_jac(subg1, subg2):
    set1 = set(subg1.keys())
    set2 = set(subg2.keys())
    t_intersection = len(set1.intersection(set2))
    t_union = len(set1.union(set2))
    return 1. * t_intersection/(t_union+1)
def build_paths(g):
    seen = dict()
    for node_source,att in g.nodes(data=True):
        list_of_edges = g.edges(node_source, data=True)
        for _,node_target,att_target in list_of_edges:
            if node_source != node_target:
                for _,node_to_compare,att_to_compare in g.edges(node_target, data=True):
                    if node_source != node_to_compare :
                        if (node_source,node_to_compare) not in seen:
                            seen[(node_source,node_to_compare)] = att_target['weight'] + att_to_compare['weight']
                        else:
                            seen[(node_source,node_to_compare)] += att_target['weight'] + att_to_compare['weight']
    return seen
class BoTG(BaseEstimator, TransformerMixin): # based on TfidfTransformer structure
    def __init__(self, eps=0.8, w=2, njobs=12):
        self.eps = eps
        self.w = w
        self.njobs = njobs
        self._idf_ = {}
    def fit(self, X, y=None, verbose=False, **fit_params):
        Gs = [ doc.G for doc in Document.build_docs(X, w=self.w) ]
        self._N = len(Gs)
        LGs = self._build_LG_(Gs)
        list_of_valids = [ (g, idx, term) for (term, (g,idx)) in list(LGs.items()) if len(idx) > 1 ]
        random.shuffle(list_of_valids)
        results = list(map( self.__build_representation_graph__, tqdm(list_of_valids, smoothing=0, disable=not verbose) ))
        #with pool.Pool(self.njobs) as p:
            #results = list(p.imap( __build_representation_graph__, tqdm(list_of_valids, disable=not verbose) ))
        self._build_model_(results)
    def transform(self, X, verbose=False):
        Gs = [ doc.G for doc in Document.build_docs(X, w=self.w) ]
        M = lil_matrix( (len(Gs), self.k) )
        for docid, G in tqdm(enumerate(Gs), disable=not verbose):
            for tid, term in enumerate(G.nodes):
                if term not in self._model_:
                    continue
                for cluster_id, cluster in self._model_[term].items():
                    weight = metric( G.neighbors(term), cluster.neighbors(term) )
                    M[docid,cluster_id] = weight*np.log( self._N/self._idf_[term] )
        return M
    def _build_model_(self, old_dict):
        self.k = 0
        self._model_ = {}
        for (term, rep) in old_dict:
            self._model_[term] = {}
            for idx_cluster, cluster_repr in rep.items():
                self._model_[term][self.k] = cluster_repr
                self.k += 1
        return self._model_, k
    def _build_LG_(self, Gs, verbose):
        LGs = {}
        for docid,G in tqdm(enumerate(Gs), total=len(Gs), disable=not verbose):
            for v in G.nodes:
                if v not in LGs:
                    self._idf_[v] = set()
                    LGs[v] = nx.Graph()
                LGs[v].append(docid)
                self._idf_[v].add(docid)
                LG_v = LGs[v]
                all_edges = list(G.out_edges(v, data=True))
                all_edges.extend( list(G.in_edges(v, data=True)) )
                for i,(s1,t1, att1) in enumerate(all_edges):
                    if (s1,t1) not in LG_v.nodes:
                        LG_v.add_node( (s1,t1), count=0 )
                    LG_v.node[(s1,t1)]['count'] += 1
                    for (s2,t2, att2) in all_edges[(i+1):]:
                        if (s2,t2) not in LG_v.nodes:
                            LG_v.add_node( (s2,t2), count=0 )
                        if ( (s1,t1),(s2,t2) ) not in LG_v.edges:
                            LG_v.add_edge( (s1,t1),(s2,t2), count=0, sum=0 )
                        LG_v.edges[(s1,t1),(s2,t2)]['count'] += 1
                        LG_v.edges[(s1,t1),(s2,t2)]['sum']   += abs( att1['weight']-att2['weight'] )
                LGs[v] = LG_v
        for v, G in tqdm(LGs.items(), total=len(LGs), disable=not verbose):
            to_remove = []
            for s,t,att in G.edges(data=True):
                if att['count'] < 2:
                    to_remove.append((s,t))
            for s,t in to_remove:
                G.remove_edge(s,t)
        return LGs
    def __build_representation_graph__(self, x):
        (g, idx, term) = x
        for s,t,att in g.edges(data=True):
            g.edges[ (s,t) ]['weight'] = att['count']/( g.degree(s) + g.degree(t) + 1 )
        seen = build_paths(g)
        mmax = max(1,int(self.eps*len(seen)))
        top_seen = sorted( seen.items(), key=lambda x: x[1], reverse=True )[:mmax]

        top_seen = [ (((source,g[source]), (target,g[target])),weight) for ((source, target),weight) in top_seen ]

        best_repr = list(map( co_app2, top_seen))
        final_graph = nx.Graph()
        for ((s,t), weight) in best_repr:
            final_graph.add_edge( s,t, weight=1./(weight+1.) )
        nodeslist=final_graph.nodes
        if len(nodeslist) == 0 or len(final_graph.edges) == 0:
            return term,{0: g}
        M = nx.to_scipy_sparse_matrix(final_graph, weight='weight', nodelist=nodeslist)
        if M.shape[0] == 0 or M.shape[1] == 0:
            return term,{0: g}
        nclusters = max(1,int(np.log(M.shape[0])))
        clustering = KMeans(n_clusters=nclusters, n_jobs=self.njobs)
        labels = clustering.fit_predict(M)
        clustered_graph = {}
        for (l, (s,t)) in list(zip(labels,nodeslist)):
            if l not in clustered_graph:
                clustered_graph[l] = nx.Graph()
            clustered_graph[l].add_edge(s,t)
        return term, clustered_graph 
        