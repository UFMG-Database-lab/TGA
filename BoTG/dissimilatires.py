import numpy as np

def dissimilarity_node(GA, GB, term):
    neighborsGA = GA[term]
    neighborsGB = GB[term]
    weightA = GA.node[term]['weight']
    weightB = GB.node[term]['weight']
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
    all_terms_union = set(neighborsGA.keys()).union(set(neighborsGB.keys()))

    dist = abs(weightA-weightB)
    
    for term_2 in all_terms_union:
        if term_2 in neighborsGA and term_2 in neighborsGB:
            dist += abs( neighborsGA[term_2]['weight'] - neighborsGB[term_2]['weight'] )
        else:
            dist += 1.
    return dist / (len(all_terms_union)+1)

def dissimilarity_row(A,B):
    _A = A[1:]
    _B = B[1:]
    wA = A[0]
    wB = B[0]
    # Tem valores não NaN
    was_value_A = np.logical_not(_A == 0)
    was_value_B = np.logical_not(_B == 0)
    or_values   = np.logical_or(was_value_A, was_value_B)  # conjunto de todos os termos q coocorem

    was_value_A = was_value_A[or_values]
    was_value_B = was_value_B[or_values]

    and_values  = np.logical_and(was_value_A, was_value_B) # conjunto que ambos termos tem
    xor_values  = np.logical_xor(was_value_A, was_value_B) # contar o número de vezes q um tem valor e o outro não

    base = abs(wA-wB)
    diff_values = sum(abs(_A[and_values]-_B[and_values]))
    unmerged_values = sum(xor_values)
    dist = base + diff_values + unmerged_values
    iters = sum(or_values)+1 # Tamanho da união dos conjutos de termos de co-ocorrencia
    return dist/iters
