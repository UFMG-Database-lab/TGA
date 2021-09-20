import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from multiprocessing import Pool
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import networkx as nx
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from nltk.corpus import stopwords as stopwords_by_lang

import copy

from tqdm.notebook import tqdm

import re
from collections import Counter
import scipy.sparse as sp
import numpy as np

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
    def __init__(self, mindf=2, lan='english', stopwords='nltk', model='topk', k=500, verbose=False):
        super(Tokenizer, self).__init__()
        self.mindf = mindf
        self.le = LabelEncoder()
        self.verbose = verbose
        self.lan = lan
        if stopwords is None:
            self.stopwordsSet = []
        elif stopwords == 'nltk':
            self.stopwordsSet = stopwords_by_lang.words(lan)
        elif stopwords == 'scikit':
            self.stopwordsSet = stop_words
        else:
            self.stopwordsSet = stopwords
        self.model =  model
        self.k     = k
        self.analyzer = TfidfVectorizer(preprocessor=preprocessor, min_df=mindf)#.build_analyzer()
        self.local_analyzer = self.analyzer.build_analyzer()
        self.analyzer.set_params( analyzer=self.local_analyzer )
        self.node_mapper      = {}
        
    def analyzer_doc(self, doc):
        return self.local_analyzer(doc)
    def fit(self, X, y):
        self.N = len(X)
        y = self.le.fit_transform( y )
        self.n_class = len(self.le.classes_)
        docs_in_terms = []
        
        with Pool(processes=18) as p:
            #docs = map(self.local_analyzer, X)
            for doc_in_terms in tqdm(p.imap(self.analyzer_doc, X), total=self.N, disable=not self.verbose):
                doc_in_terms = list(set(map( self._filter_fit_, list(doc_in_terms) ))) 
                docs_in_terms.extend(doc_in_terms)
        
        self.term_freqs       = Counter(docs_in_terms)
        self.term_freqs       = { term:v for (term,v) in self.term_freqs.items() if v >= self.mindf }
        self.node_mapper      = { term: self.node_mapper.setdefault(term, len(self.node_mapper)+1)
                                 for term in self.term_freqs.keys() }
        self.node_mapper['<BLANK>'] = 0
        self.term_freqs['<BLANK>']  = self.N
        
        self.node_mapper['<UNK>']   = len(self.node_mapper)
        self.term_freqs['<UNK>']  = self.N
        self.vocab_size = len(self.node_mapper)
        
        self.term_array = [ term for (term,term_id) in sorted(self.node_mapper.items(), key=lambda x: x[1]) ]
        
        self.fi_ = np.array([ np.log2( (self.N+1)/(self.term_freqs[term]+1) ) for term in self.term_array ])
            
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

    
class AttentionTFIDF(nn.Module):
    def __init__(self, vocab_size, hiddens, nclass, maxF=20, drop=.5,
                 initrange=.5, negative_slope=99.):
        super(AttentionTFIDF_V1, self).__init__()
        self.hiddens        = hiddens
        self.maxF           = maxF
        self.value_emb      = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.query_emb      = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.key_emb        = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.TF_emb         = nn.Embedding(maxF, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.DF_emb         = nn.Embedding(maxF, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.fc             = nn.Linear(hiddens, nclass)
        self.initrange      = initrange 
        self.negative_slope = negative_slope
        self.drop_          = drop
        self.init_weights()
    def forward(self, doc_tids, TFs, DFs):
        batch_size = doc_tids.size(0)
        bx_packed  = doc_tids == 0
        pad_mask   = bx_packed.logical_not()
        doc_sizes  = pad_mask.sum(dim=1).view(batch_size, 1)
        pad_mask   = pad_mask.view(*bx_packed.shape, 1)
        pad_mask   = pad_mask.logical_and(pad_mask.transpose(1, 2))
        
        TFs     = torch.clamp( TFs, max=self.maxF-1 )
        h_TFs   = self.TF_emb( TFs )
        h_TFs   = F.dropout( h_TFs, p=self.drop_, training=self.training )
        
        DFs     = torch.clamp( DFs, max=self.maxF-1 )
        h_DFs   = self.DF_emb( DFs )
        h_DFs   = F.dropout( h_DFs, p=self.drop_, training=self.training )
        
        h_query = self.query_emb( doc_tids )
        h_query = h_query + h_TFs + h_DFs
        #h_query = torch.tanh( h_query )
        h_query = F.dropout( h_query, p=self.drop_, training=self.training )
        
        h_key = self.key_emb( doc_tids )
        h_key = h_key + h_TFs + h_DFs
        #h_key = torch.tanh( h_key )
        h_key = F.dropout( h_key, p=self.drop_, training=self.training )
        
        co_weights  = torch.bmm( h_key, h_query.transpose( 1, 2 ) )
        #co_weights = torch.tanh( co_weights )
        #co_weights  = co_weights / torch.pow(1.+co_weights, 2.)
        co_weights  = F.leaky_relu( co_weights, negative_slope=self.negative_slope)
        
        #co_weights[pad_mask.logical_not()] = 0. # Set the 3D-pad mask values to
        #co_weights = torch.tanh(co_weights)
        
        co_weights[pad_mask.logical_not()] = float('-inf') # Set the 3D-pad mask values to -inf (=0 in sigmoid)
        co_weights = torch.sigmoid(co_weights)
        
        weights = co_weights.sum(axis=2) / doc_sizes
        weights[bx_packed] = float('-inf') # Set the 2D-pad mask values to -inf  (=0 in softmax)
        
        weights = torch.softmax(weights, dim=1)
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        weights = weights.view( *weights.shape, 1 )
        
        h_value = self.value_emb( doc_tids )
        h_value = h_value + h_TFs + h_DFs
        h_value = F.dropout( h_value, p=self.drop_, training=self.training )
        
        docs_h = h_value * weights
        docs_h = docs_h.sum(axis=1)
        docs_h = F.dropout( docs_h, p=self.drop_, training=self.training )
        docs_h = self.fc(docs_h)
        return docs_h, weights, co_weights
    
    def init_weights(self):
        self.TF_emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.DF_emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.query_emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.key_emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.value_emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.fc.weight.data.uniform_(-self.initrange, self.initrange)


class AttentionTFIDFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hiddens, mindf=2, lan='english', stopwords='nltk', k=256,
                 maxF=20, drop=.8, initrange=.5, negative_slope=99., _device=torch.device('cuda:0'),
                 verbose=False):
        super(AttentionTFIDFClassifier, self).__init__()
        self._model         = None
        self._tokenizer     = None
        self.hiddens        = hiddens
        self.mindf          = mindf
        self.lan            = lan
        self.stopwords      = stopwords
        self.maxF           = maxF
        self.k              = k
        self.drop           = drop
        self.initrange      = initrange
        self.negative_slope = negative_slope
        self.verbose        = verbose
        self._device        = _device
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        
        def collate_train(param):
            X, y = zip(*param)
            doc_tids, TFs, DFs = self._tokenizer.transform(X, verbose=self.verbose)
            
            doc_tids = pad_sequence(list(map(torch.LongTensor, doc_tids)), batch_first=True, padding_value=0)

            TFs = pad_sequence(list(map(torch.tensor, TFs)), batch_first=True, padding_value=0)
            TFs = torch.LongTensor(torch.log2(TFs+1).round().long())

            DFs = pad_sequence(list(map(torch.tensor, DFs)), batch_first=True, padding_value=0)
            DFs = torch.LongTensor(torch.log2(DFs+1).round().long())

            return doc_tids, TFs, DFs, torch.LongTensor(y)
        
        self.collate = collate_train
        
    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            pass
        self._tokenizer = Tokenizer(mindf=self.mindf, lan=self.lan, stopwords=self.stopwords,
                                   model='sample', k=self.k, verbose=self.verbose)
        self._tokenizer.fit(X, y)
        
        self._model     = AttentionTFIDF( vocab_size=self._tokenizer.vocab_size, hiddens=self.hiddens,
                                        nclass=self._tokenizer.n_class, maxF=self.maxF, drop=self.drop,
                                        initrange=self.initrange, negative_slope=self.negative_slope )
        
        
        
        optimizer = optim.AdamW( self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_func_cel = nn.CrossEntropyLoss().to( self._device )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.95,
                                                       patience=10, verbose=True)
        
        
        return self
    def predict(self, X):
        if self._model is None or self._tokenizer is None:
            raise Exception("Model is None!")
