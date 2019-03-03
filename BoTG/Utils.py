"""
Created on Mon Nov 19 11:38:02 2018

@author: Vítor Mangaravite
"""
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn import cluster

import numpy as np

import gc

from tqdm import tqdm

import time
import networkx as nx

import multiprocessing
from multiprocessing import Process

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

def garbage_collector():
    while True:
        gc.collect()
        time.sleep(5)

def K(x, sigma=100.):
    return np.exp( -(np.power(x,2.)/(2.*np.power(sigma,2.))) ) / (sigma*np.sqrt(2*np.pi))        

def _join_graph_( G, g):
        G.add_edges_from( g.edges )
        for (source,target,data) in g.edges(data=True):
            for att,value in data.items():
                if att not in G[source][target]:
                    G[source][target][att] = 0.
                G[source][target][att] += value

        for n in g.nodes:
            if n not in G:
                G.add_node(n)
            for att in g.node[n]:
                if att not in G.node[n]:
                    G.node[n][att] = 0.
                G.node[n][att] += g.node[n][att]
        return G

def _norm_graph_( G): # const-kernel
    if not len(G.nodes):
        maxWeight_vertex = 1.
    else:
        maxWeight_vertex = max([ v_attr['weight'] for _,v_attr in G.nodes(data=True) ])
    for v, v_attr in G.nodes(data=True):
        G.node[v]['weight'] = v_attr['weight'] / maxWeight_vertex
    
    if not len(G.edges):
        maxWeight_edge = 1.
    else:
        maxWeight_edge = max([ e_attr['weight'] for _,_,e_attr in G.edges(data=True) ])
    for s,t,e_attr in G.edges(data=True):
        G[s][t]['weight'] = e_attr['weight'] / maxWeight_edge

    return G

def size_item(item, size_float):
    term, docs_within = item
    #return len(docs_within)*len(docs_within)*size_float
    return 8*(len(docs_within)*len(docs_within)*size_float + 2*sys.getsizeof(docs_within))

def process_term_func(params):
    recv_end, send_end = multiprocessing.Pipe()
    p = Process(target=process_term_thread, args=(params, send_end))
    p.start()
    p.join()
    p.terminate()
    del p
    return recv_end.recv()

def process_term_func2(params):
    term, docs_within, quantile, metric, dissimilarity_func, get_subgraph, pool, verbose = params
    params2 = (term, docs_within, quantile, metric, dissimilarity_func, get_subgraph, verbose)
    for result in pool.imap( process_term, params2 ):
        return result

def process_term_thread(params, send_end):
    send_end.send(process_term(params))

def process_term(params):
    term, docs_within, quantile, metric, dissimilarity_func, get_subgraph, verbose = params
    docs_within = list(docs_within)
    M = np.zeros((len(docs_within), len(docs_within)), dtype=np.float)
    #qtd_total = int((len(docs_within)*len(docs_within))/2 - len(docs_within))
    with tqdm(total=len(docs_within), position=2, desc="Building Distances", disable=not verbose, smoothing=0.8) as pbar:
        for i, doc_i in enumerate(docs_within):
            j = i+1
            values = np.array([ 1.-dissimilarity_func(doc_i.G, doc_j.G, term) for doc_j in docs_within[j:] ])
            M[i,j:] = values
            M[j:,i] = values
            pbar.update()
            #pbar.update(len(docs_within)-j-1)
    #M = NearestNeighbors(metric=metric).fit(M).radius_neighbors_graph(mode='distance')
    eps = quantile
    #if len(M.nonzero()[0]) > 0:
    #    eps = np.percentile(M[M.nonzero()].ravel(), q=quantile)
    min_samples = int(np.sqrt(M.shape[0]))
    clusters = cluster.DBSCAN(n_jobs=1, eps=eps, min_samples=min_samples, metric=metric).fit_predict(M)
    del M
    _clusters = []
    mapper = [ [] for i in range(max(clusters)+1) ]
    list(map(lambda x: mapper[x[1]].append(docs_within[x[0]].G), [ (i,x) for (i,x) in enumerate(clusters) if x >=0 ]))
    for i, doc_in_cluster in enumerate(mapper):
        # map subgraphs
        subgraphs = list(map(lambda x: get_subgraph(x, term) , doc_in_cluster))
        # reduce subgraphs
        selected_subgraph = nx.DiGraph()
        for subgraph in subgraphs:
            selected_subgraph = _join_graph_(selected_subgraph, subgraph)
        selected_subgraph = _norm_graph_(selected_subgraph)
        _clusters.append( selected_subgraph )
        del subgraphs
    return term, _clusters
