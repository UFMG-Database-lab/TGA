from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords as stopwords_by_lang
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from scipy.sparse import csr_matrix

import re
from collections import Counter

from tqdm import tqdm

import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot


def ig(X, y):

    def get_t1(fc, c, f):
        t = np.log2(fc/(c * f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fc, t)

    def get_t2(fc, c, f):
        t = np.log2((1-f-c+fc)/((1-c)*(1-f)))
        t[~np.isfinite(t)] = 0
        return np.multiply((1-f-c+fc), t)

    def get_t3(c, f, class_count, observed, total):
        nfc = (class_count - observed)/total
        t = np.log2(nfc/(c*(1-f)))
        t[~np.isfinite(t)] = 0
        return np.multiply(nfc, t)

    def get_t4(c, f, feature_count, observed, total):
        fnc = (feature_count - observed)/total
        t = np.log2(fnc/((1-c)*f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fnc, t)

    X = check_array(X, accept_sparse='csr')
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    # counts

    observed = safe_sparse_dot(Y.T, X)          # n_classes * n_features
    total = observed.sum(axis=0).reshape(1, -1).sum()
    feature_count = X.sum(axis=0).reshape(1, -1)
    class_count = (X.sum(axis=1).reshape(1, -1) * Y).T

    # probs

    f = feature_count / feature_count.sum()
    c = class_count / float(class_count.sum())
    fc = observed / total

    # the feature score is averaged over classes
    scores = (get_t1(fc, c, f) +
            get_t2(fc, c, f) +
            get_t3(c, f, class_count, observed, total) +
            get_t4(c, f, feature_count, observed, total)).mean(axis=0)

    scores = np.asarray(scores).reshape(-1)

    return scores
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

replace_patterns = [
    ('<[^>]*>', ''),                                    # remove HTML tags
    ('(\D)\d\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D)\d\d\d\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D\D)\d\d\d\D\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D)\d\d\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedZipcodePlusFour \\2'),
    ('(\D)\d(\D)', '\\1ParsedOneDigit\\2'),
    ('(\D)\d\d(\D)', '\\1ParsedTwoDigits\\2'),
    ('(\D)\d\d\d(\D)', '\\1ParsedThreeDigits\\2'),
    ('(\D)\d\d\d\d(\D)', '\\1ParsedFourDigits\\2'),
    ('(\D)\d\d\d\d\d(\D)', '\\1ParsedFiveDigits\\2'),
    ('(\D)\d\d\d\d\d\d(\D)', '\\1ParsedSixDigits\\2'),
    ('\d+', 'ParsedDigits')
]

compiled_replace_patterns = [(re.compile(p[0]), p[1]) for p in replace_patterns]

def generate_preprocessor(replace_patterns):
    compiled_replace_patterns = [(re.compile(p[0]), p[1]) for p in replace_patterns]
    def preprocessor(text):
        for pattern, replace in compiled_replace_patterns:
            text = re.sub(pattern, replace, text)
        text = text.lower()
        return text
    return preprocessor

generated_patters=generate_preprocessor(replace_patterns)

def preprocessor(text):
    # For each pattern, replace it with the appropriate string
    for pattern, replace in compiled_replace_patterns:
        text = re.sub(pattern, replace, text)
    text = text.lower()
    return text

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, mindf=2, lan='english', stopwordsSet='nltk', model='topk', k=500,
                 vocab_max_size=99999999999, ngram_range=(1,1), verbose=False):
        super(Tokenizer, self).__init__()
        self.mindf = mindf
        self.le = LabelEncoder()
        self.verbose = verbose
        self.lan = lan
        if stopwordsSet == 'nltk':
            self.stopwordsSet = stopwords_by_lang.words(lan)
        elif stopwordsSet == 'scikit':
            self.stopwordsSet = stop_words
        else:
            self.stopwordsSet = []
        self.model =  model
        self.k     = k
        self.ngram_range = ngram_range
        self.analyzer = TfidfVectorizer(ngram_range=ngram_range,stop_words=self.stopwordsSet,
                                        max_features=vocab_max_size,
                                        preprocessor=preprocessor, min_df=mindf)#.build_analyzer()
        self.local_analyzer = self.analyzer.build_analyzer()
        self.analyzer.set_params( analyzer=self.local_analyzer )
        self.node_mapper      = {}
        self.vocab_max_size   = vocab_max_size
        
    def analyzer_doc(self, doc):
        return self.local_analyzer(doc)
    def fit(self, X, y):
        self.N           = len(X)
        y                = self.le.fit_transform( y )
        self.n_class     = len(self.le.classes_)
        docs_in_tids     = []
        self.node_mapper = {}
        
        with Pool(processes=64) as p:
            for docid,doc_in_terms in tqdm(enumerate(p.imap(self.analyzer_doc, X)),
                                           total=self.N, disable=not self.verbose):
                doc_in_terms = list(set(map( self._filter_fit_, list(doc_in_terms) ))) 
                docs_in_tids.extend( [ (1,docid,term)
                                      for term in doc_in_terms ] )
        data, row_ind, docs_in_terms = list(zip(*docs_in_tids))
        self.term_freqs              = Counter(docs_in_terms)
        self.term_freqs              = { term: freq for term,freq in self.term_freqs.items() if freq >= self.mindf  }
        docs_in_tids                 = [ (d,docid,term) for (d,docid,term) in docs_in_tids if term in self.term_freqs ]
        data, row_ind, docs_in_terms = list(zip(*docs_in_tids))
        
        
        col_ind                      = [ self.node_mapper.setdefault(term, len(self.node_mapper))
                                        for term in docs_in_terms ]
        
        X_vec = csr_matrix((data, (row_ind, col_ind)), shape=(self.N, len(self.term_freqs)))
        
        self.term_array = [ term for (term,term_id) in sorted(self.node_mapper.items(), key=lambda x: x[1]) ]
        
        res = list(zip(self.term_array, ig(X_vec, y)))
        res = sorted(res, key=lambda x: x[1], reverse=True)[:self.vocab_max_size]
        
        self.term_freqs             = { term:self.term_freqs[term] for term, weight in res }
        self.maxF                   = np.max( np.array(list(self.term_freqs.values())) )
        self.term_freqs['<BLANK>']  = self.N
        self.term_freqs['<UNK>']    = self.N
        
        self.node_mapper = { term: termid for (termid, (term, weight)) in zip(range(len(res)), res) }
        self.node_mapper['<BLANK>'] = len(self.node_mapper)
        self.node_mapper['<UNK>']   = len(self.node_mapper)
        self.term_array = [ term for (term,term_id) in sorted(self.node_mapper.items(), key=lambda x: x[1]) ]
        
        
        self.fi_ = [ weight for term, weight in res ]
        self.fi_.append(0.)
        self.fi_.append(0.)
        self.fi_ = np.array(self.fi_)
        
        self.vocab_size = len(self.node_mapper)
        return self
    def _filter_transform_(self, term):
        if term in self.stopwordsSet:
            return '<STPW>'
        if term not in self.node_mapper:
            return '<UNK>'
        return term
    def _filter_fit_(self, term):
        if term in self.stopwordsSet:
            return '<STPW>'
        return term
    def _model_(self, doc):
        doc_counter = Counter(doc)
        doc = np.array(list(doc_counter.keys()))
        if len(doc) > self.k:
            weigths = np.array([ self.fi_[t] for t in doc ])
            weigths = softmax(weigths)
            if self.model == 'topk':
                doc = doc[(-weigths).argsort()[:self.k]]
            elif self.model == 'sample':
                doc = np.random.choice(doc, size=self.k, replace=False, p=weigths)
        TFs = np.array([ doc_counter[tid] for tid in doc ])
        DFs = np.array([ self.term_freqs[self.term_array[tid]] for tid in doc ])
        return doc, TFs, DFs
    def transform(self, X, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        n = len(X)
        terms_ = []
        for i,doc_in_terms in tqdm(enumerate(map(self.analyzer_doc, X)), total=n, disable=not verbose):
            doc_in_terms = map( self._filter_transform_, doc_in_terms )
            #doc_in_terms = filter( lambda x: x != '<STPW>', doc_in_terms )
            doc_tids = [ self.node_mapper[tid] for tid in doc_in_terms ]
            doc_tids, TFs, DFs = self._model_(doc_tids)
            terms_.append( (doc_tids, TFs, DFs) )
        doc_tids, TFs, DFs = list(zip(*terms_))
        return list(doc_tids), list(TFs), list(DFs)