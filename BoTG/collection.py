class Collection(object):
    def __init__(self, docspath=None, minDF=2.,
    n_jobs=None, max_iter=100, metric='cosine',bandwidth=None,quantile=0.001, proportion=30,
    pooling='mean', assignment='hard',
    **kwargs):
        self.documents = dict()
        self.terms_idx = dict()
        self.minDF = minDF
        self.n_jobs = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
        #if self.n_jobs is None:
        #    self.n_jobs = multiprocessing.cpu_count()
        self.max_iter = max_iter
        self.metric = metric
        self.bandwidth = bandwidth
        self.proportion = proportion
        self.quantile = quantile
        self.seeds = None

        self.pooling_mode = pooling
        self.__pooling__ = self.__get_pooling__()

        self.assignment_mode = assignment
        self.__assignment__ = self.__get_assignment__()

        if docspath is not None:
            self.load_path(docspath, **kwargs)
    def add_doc(self, doc):
        self.documents[doc.id] = doc
        avgTF = np.mean([ node[1]['TF'] for node in doc.G.nodes(data=True)])
        stdTF = np.std([ node[1]['TF'] for node in doc.G.nodes(data=True)])
        for v_term in [ node[0] for node in doc.G.nodes(data=True) if node[1]['TF']/(avgTF+stdTF) >= 1. ]:
            if v_term not in self.terms_idx:
                self.terms_idx[v_term] = set()
            self.terms_idx[v_term].add( doc.id )
    def add_docs(self, docs):
        list(map(self.add_doc, docs))
    def load_path(self, docspath, **kwargs):
        if not docspath.endswith('*'):
            docspath = path.join(docspath, '*')
        all_files = glob(docspath)
        self.add_docs(tqdm(map(lambda x: Document.load_document(x, **kwargs), all_files), total=len(all_files), desc='Loading path'))
        """ # This parallel version must be fixed
        with Pool(4) as pool:
            with tqdm(total=len(all_files)) as pbar:
                for document in tqdm(pool.imap_unordered(lambda x: LoadDocument(x, **kwargs), all_files)):
                    self.add_doc(document)
                    pbar.update()
        """
    def build_matrix_new(self):
        self.terms_doc_arr = [ (term, docid) for term in self.terms_idx for docid in self.terms_idx[term] if len(self.terms_idx[term]) >= self.minDF  ] #$
        self.matrix_term_doc_idx = {}
        for i, (term, doc) in tqdm(enumerate(self.terms_doc_arr), total=len(self.terms_doc_arr), desc="Term-Doc Index"):
            if term not in self.matrix_term_doc_idx:
                self.matrix_term_doc_idx[term] = {}
            self.matrix_term_doc_idx[term][doc] = i

        self.X = scsr_matrix((len(self.terms_doc_arr),len(self.terms_doc_arr)), dtype=np.float)
        #self.X.setdiag(np.ones(len(self.terms_doc_arr)))

        values_diss = []
        doc_term_idxs_A = []
        doc_term_idxs_B = []

        qtd_comps = int(sum([ (len(docs)*(len(docs)-1))/2 for docs in self.matrix_term_doc_idx.values() ]))
        with tqdm(total=qtd_comps, desc='Term distances') as pbar:
            for term in self.matrix_term_doc_idx:
                docs = sorted( self.matrix_term_doc_idx[term].items(), key=lambda x: x[1] ) # (doc,idx_matrix)
                for i, (docid_1, doc_term_id_1) in enumerate(docs):
                    for (docid_2, doc_term_id_2) in docs[:i]:
                        sim = self.__dissimilarity_node__( self.documents[docid_1].G, self.documents[docid_2].G, term )
                        #self.X[doc_term_id_1,doc_term_id_2] = 1.-sim
                        values_diss.append(1.-sim)
                        doc_term_idxs_A.append(doc_term_id_1)
                        doc_term_idxs_B.append(doc_term_id_2)
                        pbar.update()
        print("Creating diagonal", flush=True)
        self.X.setdiag(1.)
        print("Setting values", flush=True)
        self.X[doc_term_idxs_A,doc_term_idxs_B] = values_diss
    def build_matrix(self):
        self.terms_doc_arr = [ (term, docid) for term in self.terms_idx for docid in self.terms_idx[term] if len(self.terms_idx[term]) > self.minDF  ] # (doc.id,v_term)
        self.matrix_term_doc_idx = {}
        for i, (term, doc) in tqdm(enumerate(self.terms_doc_arr), total=len(self.terms_doc_arr), desc="Term-Doc Index"):
            if term not in self.matrix_term_doc_idx:
                self.matrix_term_doc_idx[term] = {}
            self.matrix_term_doc_idx[term][doc] = i
        
        #self.X = scipy.sparse.lil_matrix((len(self.terms_doc_arr),len(self.terms_doc_arr)), dtype=np.float)
        #self.X.setdiag(np.ones(len(self.terms_doc_arr)))

        values_diss = []
        doc_term_idxs_A = []
        doc_term_idxs_B = []

        qtd_comps = int(sum([ (len(docs)*(len(docs)-1))/2 for docs in self.matrix_term_doc_idx.values() ]))
        with tqdm(total=qtd_comps, desc='Term distances') as pbar:
            for term in self.matrix_term_doc_idx:
                docs = sorted( self.matrix_term_doc_idx[term].items(), key=lambda x: x[1] ) # (doc,idx_matrix)
                for i, (docid_1, doc_term_id_1) in enumerate(docs):
                    for (docid_2, doc_term_id_2) in docs[i+1:]:
                        sim = self.__dissimilarity_node__( self.documents[docid_1].G, self.documents[docid_2].G, term )
                        values_diss.append(1.-sim)
                        doc_term_idxs_A.append(doc_term_id_1)
                        doc_term_idxs_B.append(doc_term_id_2)
                        pbar.update()
        print()
        print("Creating sparse matrix", flush=True)
        self.X = scipy.sparse.eye(len(self.terms_doc_arr), format='csr')
        print("Assigment axis=0", flush=True)
        self.X[doc_term_idxs_A,doc_term_idxs_B] = values_diss
        print("Assigment axis=1", flush=True)
        self.X[doc_term_idxs_B,doc_term_idxs_A] = values_diss

    def build_vocabulary(self):
        #self.X_csr = self.X.tocsr()
        result_dict = build_clusters(self.X, self.n_jobs, self.max_iter, self.bandwidth, self.seeds, self.metric, self.proportion)
        self.clusters = result_dict['clusters']
        self.mapper_subgraph_cluster = result_dict['mapper_cluster']
        self.bandwidth = result_dict['bandwidth']
        self.idx_term_cluster = [ self.terms_doc_arr[idx] for idx in sorted(list(self.clusters.keys())) ]
        self.terms_in = {}
        for i, (term,_) in enumerate(self.idx_term_cluster):
            if term not in self.terms_in:
                self.terms_in[term] = []
            self.terms_in[term].append(i)
    def __assignment_hard__(self, doc):
        result = scipy.sparse.lil_matrix((len(doc.G.nodes),len(self.idx_term_cluster)), dtype=np.float)
        for i, term in enumerate(doc.G.nodes):
            if term in self.terms_in:
                j = min(self.terms_in[term], key=lambda x: self.__dissimilarity_node__(self.documents[self.idx_term_cluster[x][1]].G, doc.G, term) )
                result[i,j] = 1.
        doc.assignment = result
        return result
    def __assignment_soft__(self, doc):
        result = np.ones((len(doc.G.nodes),len(self.idx_term_cluster)))
        for i, term in enumerate(doc.G.nodes):
            if term in self.terms_in:
                for j in self.terms_in[term]:
                    idx_document = self.idx_term_cluster[j][1]
                    result[i,j] = self.__dissimilarity_node__(self.documents[idx_document].G, doc.G, term)
        result = K(result)
        result = result / result.sum(axis=1, keepdims=True)
        doc.assignment =  np.matrix(result)
        return doc.assignment
    def __get_assignment__(self, assignment_mode=None):
        assignment_name = self.assignment_mode
        if assignment_mode is not None:
            assignment_name = assignment_mode
        if assignment_name == 'hard':
            return self.__assignment_hard__
        elif assignment_name == 'soft':
            return self.__assignment_soft__
        else:
            raise ValueError("%s does not available as assignment function" % assignment_name)
    def __pooling_mean__(self, doc):
        doc.pooling = np.mean(doc.assignment, axis=0).A[0]
        return doc.pooling
    def __pooling_max__(self, doc):
        doc.pooling = doc.assignment.tocsr().max(axis=0).A[0]
        return doc.pooling
    def __pooling_sum__(self, doc):
        doc.pooling = np.sum(doc.assignment, axis=0).A[0]
        return doc.pooling
    def __get_pooling__(self, pooling_mode=None):
        pooling_name = self.pooling_mode
        if pooling_mode is not None:
            pooling_name = pooling_mode
        if pooling_name == 'mean':
            return self.__pooling_mean__
        elif pooling_name == 'max':
            return self.__pooling_max__
        elif pooling_name == 'sum':
            return self.__pooling_sum__
        else:
            raise ValueError("%s does not available as pooling function" % pooling_name)
    
    def __dissimilarity_node__(self, GA, GB, term):
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
    def assignment(self, assignment_mode=None):
        __assignment_run__ = self.__assignment__
        if assignment_mode is not None:
            __assignment_run__ = self.__get_assignment__(assignment_mode)
        for doc in tqdm(self.documents.values(), desc="Assignment"):
            __assignment_run__(doc)
    def pooling(self, pooling_mode=None):
        __pooling_run__ = self.__pooling__
        if pooling_mode is not None:
            __pooling_run__ = self.__get_pooling__(pooling_mode)
        for doc in tqdm(self.documents.values(), desc="Pooling"):
            __pooling_run__(doc)
    def process_collection(self, pooling_mode=None, assignment_mode=None):
        self.build_matrix()
        self.build_vocabulary()
        self.assignment(assignment_mode)
        self.pooling(pooling_mode)