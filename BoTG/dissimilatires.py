import numpy as np

def dissimilarity_node_out(GA, GB, term):
    neighborsGA = GA[term]
    neighborsGB = GB[term]
    weightA = GA.node[term]['weight']
    weightB = GB.node[term]['weight']
    return __compute_diss__(neighborsGA, weightA, neighborsGB, weightB)

def dissimilarity_node_in(GA, GB, term):
    weightA = GA.node[term]['weight']
    weightB = GB.node[term]['weight']
    neighborsGA = dict([(node,D) for node,_,D in  GA.in_edges(term, data=True)])
    neighborsGB = dict([(node,D) for node,_,D in  GB.in_edges(term, data=True)])
    return __compute_diss__(neighborsGA, weightA, neighborsGB, weightB)

def dissimilarity_node_both(GA, GB, term):
    weightA = GA.node[term]['weight']
    weightB = GB.node[term]['weight']

    neighborsGA = dict([(node,D) for node,_,D in  GA.in_edges(term, data=True)])
    neighborsGB = dict([(node,D) for node,_,D in  GB.in_edges(term, data=True)])
    in_dissimilarity = __compute_diss__(neighborsGA, weightA, neighborsGB, weightB)

    neighborsGA = GA[term]
    neighborsGB = GB[term]
    out_dissimilarity = __compute_diss__(neighborsGA, weightA, neighborsGB, weightB)

    return (in_dissimilarity+out_dissimilarity)/2.

def __compute_diss__(neighborsGA, weightA, neighborsGB, weightB):
    values_A = set(neighborsGA.keys())
    values_B = set(neighborsGB.keys())
    
    size_union = len(values_A.union(values_B))
    intersection_neigh = values_A.intersection(values_B)

    dist = abs(weightA-weightB) + len( values_A ^ values_B ) # values_A ^ values_B = XOR(values_A, values_B)
    
    dist += sum( [ abs( neighborsGA[term_2]['weight'] - neighborsGB[term_2]['weight'] ) for term_2 in intersection_neigh ] )

    #for term_2 in intersection_neigh:
    #    dist += abs( neighborsGA[term_2]['weight'] - neighborsGB[term_2]['weight'] )
    
    return dist / (size_union+1)
