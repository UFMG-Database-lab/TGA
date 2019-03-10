"""
Created on Mon Nov 19 11:38:02 2018

@author: Vítor Mangaravite
"""
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn import cluster

import numpy as np

import networkx as nx

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

def _get_subgraph_out_( G, term):
        g = nx.DiGraph()
        g.add_edges_from( G.edges(term, data=True) )
        if len(g.nodes) == 0:
            # this term-node does not have any edge
            g.add_node(term)
        for n in g.nodes:
            for att in G.node[n]:
                g.node[n][att] = G.node[n][att]
        return g
def _get_subgraph_in_( G, term):
    g = nx.DiGraph()
    g.add_edges_from( G.in_edges(term, data=True) )
    if len(g.nodes) == 0:
        # this term-node does not have any edge
        g.add_node(term)
    for n in g.nodes:
        for att in G.node[n]:
            g.node[n][att] = G.node[n][att]
    return g
def _get_subgraph_both_( G, term):
    g = nx.DiGraph()
    g.add_edges_from( G.edges(term, data=True) )
    g.add_edges_from( G.in_edges(term, data=True) )
    if len(g.nodes) == 0:
        # this term-node does not have any edge
        g.add_node(term)
    for n in g.nodes:
        for att in G.node[n]:
            g.node[n][att] = G.node[n][att]
    return g