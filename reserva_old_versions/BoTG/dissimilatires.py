
def dissimilarity_node_out(GA, GB, term, term2=None):
    if term2 is None:
        term2 = term
    weightA = GA.node[term]['weight']
    weightB = GB.node[term2]['weight']

    neighborsGA = dict([((s,t),D['weight']) for s,t,D in GA.out_edges(term, data=True)])
    neighborsGB = dict([((s,t),D['weight']) for s,t,D in GB.out_edges(term2, data=True)]) 
    
    return __compute_diss__(neighborsGA, weightA, neighborsGB, weightB)

def dissimilarity_node_in(GA, GB, term, term2=None):
    if term2 is None:
        term2 = term
    weightA = GA.node[term]['weight']
    weightB = GB.node[term2]['weight']
    
    neighborsGA = dict([(node,D['weight']) for node,_,D in  GA.in_edges(term, data=True)])
    neighborsGB = dict([(node,D['weight']) for node,_,D in  GB.in_edges(term2, data=True)])

    return __compute_diss__(neighborsGA, weightA, neighborsGB, weightB)

def dissimilarity_node_both_old(GA, GB, term, term2=None):
    if term2 is None:
        term2 = term
    weightA = GA.node[term]['weight']
    weightB = GB.node[term2]['weight']

    neighborsGA = dict([((s,t),D['weight']) for s,t,D in  GA.in_edges(term, data=True)])
    neighborsGB = dict([((s,t),D['weight']) for s,t,D in  GB.in_edges(term2, data=True)])
    in_diss = __compute_diss__(neighborsGA, weightA, neighborsGB, weightB)

    neighborsGA = dict([((s,t),D['weight']) for s,t,D in  GA.out_edges(term, data=True)])
    neighborsGB = dict([((s,t),D['weight']) for s,t,D in  GB.out_edges(term2, data=True)])
    out_diss = __compute_diss__(neighborsGA, weightA, neighborsGB, weightB)

    return (in_diss+out_diss)/2.

def dissimilarity_node_both(GA, GB, term, term2=None):
    if term2 is None:
        term2 = term
    weightA = GA.node[term]['weight']
    weightB = GB.node[term2]['weight']

    neighborsGA = dict([((s,t),D['weight']) for s,t,D in  GA.in_edges(term, data=True)])
    neighborsGA.update(dict([((s,t),D['weight']) for s,t,D in  GA.out_edges(term, data=True)]))

    neighborsGB = dict([((s,t),D['weight']) for s,t,D in  GB.out_edges(term2, data=True)])
    neighborsGB.update(dict([((s,t),D['weight']) for s,t,D in  GB.in_edges(term2, data=True)]))
    return __compute_diss__(neighborsGA, weightA, neighborsGB, weightB)

def __compute_diss__(neighborsGA, weightA, neighborsGB, weightB):
    values_A = set(neighborsGA.keys())
    values_B = set(neighborsGB.keys())
    size_union = len(values_A.union(values_B))
    intersection_neigh = values_A.intersection(values_B)
    dist = abs(weightA-weightB) + len( values_A ^ values_B ) # values_A ^ values_B = XOR(values_A, values_B)
    dist += sum( [ abs( neighborsGA[term_2] - neighborsGB[term_2] ) for term_2 in intersection_neigh ] )
    return dist / (size_union+1)

    #for term_2 in intersection_neigh:
    #    dist += abs( neighborsGA[term_2]['weight'] - neighborsGB[term_2]['weight'] )
    
