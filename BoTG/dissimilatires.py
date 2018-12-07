def dissimilarity_node(GA, GB, term):
    neighborsGA = GA[term]
    neighborsGB = GB[term]
    all_terms_union = set(neighborsGA.keys()).union(set(neighborsGB.keys()))

    dist = abs(GA.node[term]['weight']-GB.node[term]['weight'])
    
    for term_2 in all_terms_union:
        if term_2 in neighborsGA and term_2 in neighborsGB:
            dist += abs( neighborsGA[term_2]['weight'] - neighborsGB[term_2]['weight'] )
        else:
            dist += 1.
    return dist / (len(all_terms_union)+1)