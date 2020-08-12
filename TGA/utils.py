import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv, GATConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from itertools import repeat
from glob import glob
from collections import namedtuple
from os import path, remove
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

from sklearn.preprocessing import LabelEncoder
import io
from nltk.corpus import stopwords as stopwords_by_lang

import re
from collections import Counter
import scipy.sparse as sp
import numpy as np

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
        # For each pattern, replace it with the appropriate string
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

WithValFold = namedtuple('Fold', ['X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val'])
Fold = namedtuple('Fold', ['X_train', 'y_train', 'X_test', 'y_test'])

class Dataset(object):
    def __init__(self, dataset_path, random_state=42, encoding='utf8'):
        super(Dataset, self).__init__()
        self.random_state = random_state
        self.encoding = encoding
        self.dataset_path = dataset_path
        self.dname = path.basename(path.abspath(dataset_path))
        self._load_dataset_()
        self._identify_splits_()
        self.nclass = len(set(self.y))
        self.split = {}
    def __str__(self):
        return f"<Dataset ({self.dname})>"
    def __repr__(self):
        return self.__str__()
    @property
    def ndocs(self):
        return len(self.y)

    @staticmethod
    def read_lines(filename):
        with io.open(filename, newline='\n') as filin:
            return filin.readlines()
        
    @staticmethod
    def get_array(X, idxs):
        return [ X[idx] for idx in idxs ]
    
    @staticmethod
    def _load_splits_(folddir, encoding='utf8'):
        splits = []
        with open(folddir, encoding=encoding, errors='ignore') as fileout:
            for line in fileout.readlines():
                fold = []
                for idx_part in line.split(';'):
                    index = list(map(int, idx_part.split()))
                    fold.append( index )
                splits.append( tuple(fold) )
        return splits
    
    def get_split(self, nfold, force_create=False, save=True, with_val=True):
        nfold = str(nfold)
        if nfold in self.split:
            folds = self.split[nfold]
            self.available_splits.add( nfold )
            if not with_val: # extends train_index with val_index
                folds = self._split_without_val_(folds)
            return folds
        
        if nfold not in self.available_splits:
            if not force_create:
                raise Exception(f"[ERROR] The {nfold}-fold split doesn't exists. Use force_create=True to create or select one available split (See the available in available_splits).")
            folds = self._create_splits_( int(nfold) )
            self.available_splits.add( nfold )
        else:
            split_file = path.join(self.dataset_path, 'splits', f'split_{nfold}.csv')
            folds = Dataset._load_splits_( split_file, self.encoding )
            
        if any([len(f) != 3 for f in folds]):
            folds = self._create_val_(folds)
            
        self.split[nfold] = folds
        
        if save:
            self.save_split(nfold)
        
        if not with_val: # extends train_index with val_index
            folds = self._split_without_val_(folds)
        
        return folds
    
    def get_fold_instances(self, nfold, force_create=True, save=True, with_val=True):
        splits = self.get_split( nfold, force_create=force_create, save=save, with_val=with_val)
        for s in splits:
            yield self._get_fold_instance_(s)
    
    def del_split(self, split, del_file=False):
        split = str(split)
        self.available_splits.remove( split )
        self.split.pop( split )
        if del_file:
            split_file = path.join(self.dataset_path, 'splits', f'split_{split}.csv')
            if path.exists( split_file ):
                remove( split_file )
    
    def save_split(self, split, force_create=True, pathtosave=None):
        splits = self.get_split( split, force_create=force_create )
        pathtosave = self.dataset_path if pathtosave is None else pathtosave
        split_file = path.join(pathtosave, 'splits', f'split_{split}.csv')
        with open(split_file, 'w', encoding=self.encoding, errors='ignore') as fileout:
            for train_index, val_index, test_index in splits:
                train_str = ' '.join(list(map(str, train_index)))
                val_str   = ' '.join(list(map(str, val_index)))
                test_str  = ' '.join(list(map(str, test_index)))
                line = train_str + ';' + val_str + ';' + test_str + '\n'
                fileout.write(line)
    
    def _create_splits_(self, k):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        kf  = list(skf.split(self.texts, self.y))
        return kf
                
    def _split_without_val_(self, splits):
        nfolds = []
        for train_index, val_index, test_index in splits:
            train_index = np.concatenate([train_index,val_index])
            nfolds.append( (train_index, test_index) )
        return nfolds
    
    def _get_fold_instance_(self, s):
        if len(s) == 2:
            train_idx, test_idx = s
            X_train = Dataset.get_array( self.texts, train_idx )
            y_train = Dataset.get_array( self.y, train_idx )
            X_test  = Dataset.get_array( self.texts, test_idx )
            y_test  = Dataset.get_array( self.y, test_idx )
            return Fold( X_train, y_train, X_test, y_test )
        elif len(s) == 3:
            train_idx, val_idx, test_idx = s
            X_train = Dataset.get_array( self.texts, train_idx )
            y_train = Dataset.get_array( self.y, train_idx )
            X_val   = Dataset.get_array( self.texts, val_idx )
            y_val   = Dataset.get_array( self.y, val_idx )
            X_test  = Dataset.get_array( self.texts, test_idx )
            y_test  = Dataset.get_array( self.y, test_idx )
            return WithValFold( X_train, y_train, X_test, y_test, X_val, y_val )
    
    def _create_val_(self, split):
        aux_split = []
        for (train_ids, test_ids) in split:
            try:
                train_idx_atual, val_idx_atual = train_test_split(train_ids,
                                                test_size=len(test_ids),
                                                stratify=Dataset.get_array(self.y, train_ids),
                                                random_state=self.random_state)
            except ValueError:
                train_idx_atual, val_idx_atual = train_test_split(train_ids,
                                                test_size=len(test_ids),
                                                random_state=self.random_state)
                
            aux_split.append( (train_idx_atual, val_idx_atual, test_ids) )
        return aux_split

    def _load_dataset_(self):
        self.texts = Dataset.read_lines(path.join(self.dataset_path, 'texts.txt'))
        self.y = Dataset.read_lines(path.join(self.dataset_path, 'score.txt'))
        self.y = list(map(int, self.y))
        
    def _identify_splits_(self):
        splits_files = glob( path.join(self.dataset_path, 'splits', 'split_*.csv') )
        self.available_splits = set(map(lambda x: path.basename(x)[6:-4], splits_files ))

class GraphsizePretrained(BaseEstimator, TransformerMixin):
    def __init__(self, w=2, pretrained_vec='glove.6B.100d', stopwords='remove', encoding='utf-8', verbose=False):
        super(GraphsizePretrained, self).__init__()
        self.w = w
        self.pretrained_vec = pretrained_vec
        self.stopwords = stopwords
        self.embeddings_dict = {}
        self.le = LabelEncoder()
        
        if not verbose:
            self.progress_bar = lambda x: x
        else:
            from tqdm import tqdm
            self.progress_bar = tqdm
        zero_based_stopword = np.array([])
        if self.stopwords == "mark":
            zero_based_stopword = np.array([0])
        
        if self.pretrained_vec.lower().endswith('.zip'):
            import zipfile
            fil = zipfile.ZipFile(self.pretrained_vec)
            f   = fil.open(fil.filelist[0].filename)
            prepro = lambda x: str(x.decode(encoding))
            fil.close()
        else:
            prepro = lambda x: x
            f = open(self.pretrained_vec, 'r')

        for line in self.progress_bar(f):
            values = line.split()
            word = prepro(values[0])
            vector = np.asarray(values[1:], "float32")
            vector = np.concatenate((vector,zero_based_stopword))
            self.ndim = len(vector) + zero_based_stopword.size
            self.embeddings_dict[word] = vector
        f.close()

        if self.stopwords == "mark":
            stopwords_list = stopwords_by_lang.words('english')
            for stp in stopwords_list:
                if stp in self.embeddings_dict:
                    self.embeddings_dict[stp][-1] = 1
        if self.stopwords == "remove":
            stopwords_list = stopwords_by_lang.words('english')
            list(map(self.embeddings_dict.pop, [stp for stp in stopwords_list if stp in self.embeddings_dict]))

        self.vocab = { word: i for (i,word) in enumerate( self.embeddings_dict.keys() ) }
        self.vocab_idx = [ k for k,v in sorted(self.vocab.items(), key=lambda x: x[1]) ]
        
        self.analyzer = TfidfVectorizer(preprocessor=preprocessor)
    
    def fit(self, X, y):
        self.N = len(X)
        y_train = self.le.fit_transform( y )
        self.n_class = len(self.le.classes_)

        self.label_ids = [ y for y in range(self.n_class) ]
        for y in self.label_ids:
            hotenc = np.zeros(self.ndim)
            hotenc[y] = 1
            self.embeddings_dict[y] = hotenc
        
        docs = list(map(self.analyzer.build_analyzer(), self.progress_bar(X)))
        edges_to_add = set( [ (y,y) for y in self.label_ids] )
        self.node_mapper = { y: y for y in self.label_ids }
        for (doc,y) in zip( docs, y_train ):

            doc_in_terms = set(filter( lambda x: x in self.embeddings_dict, doc))
            terms_by_nid = list(map(lambda x: self.node_mapper.setdefault(x, len(self.node_mapper)), doc_in_terms))

            list_of_edges = list(map( lambda x: (y, x), terms_by_nid ))
            list_of_edges.extend(list(map( lambda x: (x, x), terms_by_nid )))
            list(map(edges_to_add.add, list_of_edges))

        self.g = nx.Graph()
        self.g.add_nodes_from( [ (nid, {'idx': [nid], 'label': int(type(t) is not str), 'emb': self.embeddings_dict[t], 'term': t} ) for (t,nid) in self.node_mapper.items() ] )
        self.g.add_edges_from( edges_to_add )
        
        return self
   
    def transform(self, text):
        docs = list(map(self.analyzer.build_analyzer(), text))
        result = list(map(self._build_graph_, docs))
        return result
    
    def _build_graph_(self, doc):
        terms        = list(filter( lambda x: x in self.node_mapper, doc))
        local_mapper = { self.node_mapper[word]:word for word in set(terms) }
        terms_nids   = [ self.node_mapper[word] for word in terms ]

        cooccur_count = Counter()
        for i,nid in enumerate(terms_nids):
            terms_to_add = terms_nids[ max(i-self.w, 0):(i+1) ]
            terms_to_add = list(zip(terms_to_add, repeat(nid)))
            terms_to_add = list(map(sorted,terms_to_add))
            terms_to_add = list(map(tuple,terms_to_add))
            cooccur_count.update( terms_to_add )
        
        G = nx.Graph()
        G.add_nodes_from( [ (nid,{'term': word,'idx':[nid], 'emb': self.embeddings_dict[word]}) for (nid,word) in local_mapper.items() ] )
        w_edges = [ (s,t,w) for ((s,t),w) in cooccur_count.items() ]
        G.add_weighted_edges_from( w_edges, weight='freq' )
        
        #return G, np.array([ self.embeddings_dict[self.vocab_idx[termid]] for termid in sorted_terms ])
        return G
    
    def collate(self, param):
        X, y = zip(*param)
        Gs = self.transform(X)
        return Gs, y
class Graphsize(BaseEstimator, TransformerMixin):
    def __init__(self, lang='english', w=2, min_df=2, max_feat=999999999, feature_type='prob', stem=True, analyzer=None, verbose=False):
        super(Graphsize, self).__init__()
        self.lang = lang
        self.feature_type = feature_type
        self.w = w
        self.min_df = min_df
        self.max_feat = max_feat
        if not verbose:
            self.progress_bar = lambda x: x
        else:
            from tqdm import tqdm
            self.progress_bar = tqdm
        
        self.stopwords = set(stopwords.words('english'))
        
        self._stem_ = lambda x: x
        if stem:
            from nltk.stem.snowball import SnowballStemmer
            self._stem_ = SnowballStemmer(lang).stem
            
        self.analyzer = analyzer
        if self.analyzer is None:
            self.analyzer = TfidfVectorizer(preprocessor=preprocessor)
        
        self.vocab = dict()
        self.df = Counter()
        
    def fit(self, X, y=None):
        self.N = len(X)
        list(map(self._build_df_, self.progress_bar(X)))
        self._filter_()
        self._build_vocab_()
        return self
        
    def _build_df_(self, text):
        terms = list( filter( lambda x: x not in self.stopwords,
                             self.analyzer.build_analyzer()(text) ))
        terms = list( map( self._stem_, terms ))
        self.df.update( set(terms) )
    
    def _filter_(self):
        self.df = self.df.most_common(self.max_feat)
        self.df = dict( list(filter( lambda x: x[1] >= self.min_df, self.df)) )
    
    def _build_vocab_(self):
        self.id2term = sorted(list(self.df.keys()))
        self.vocab = dict( [ (k,i) for (i,k) in enumerate(self.id2term) ] )
        
        self.vocab['<UNK>'] = len(self.id2term)
        self.id2term.append( '<UNK>' )
   
    def transform(self, text):
        docs = list(map(self.analyzer.build_analyzer(), self.progress_bar(text)))
        result = list(map(self._build_graph_, self.progress_bar(docs)))
        result = list(map(self._build_features_, self.progress_bar(result)))
        
        return result
    
    def _prob_cooccur_(self, G, tf, sorted_nodes ):
        Adj = nx.to_scipy_sparse_matrix( G, weight='freq', nodelist=sorted_nodes )
        Adj.setdiag( tf )
        Adj = sp.csr_matrix( Adj / np.array(tf) )
        Adj.eliminate_zeros()

        #A = A.multiply(full_weight)
        row,col = Adj.nonzero()
        col2 = np.array([ sorted_nodes[c] for c in col ])

        return sp.csr_matrix( (Adj.data, (row, col2)), shape=(Adj.shape[0], len(self.vocab)) )
    def _full_weight_(self, tfidf, pg, sorted_nodes ):
        full_weight = np.array(np.array(tfidf)*np.array([pg]).T)
        row = np.array( [ [i]*len(sorted_nodes) for i in range(len(sorted_nodes)) ] )
        col = np.array( [ sorted_nodes for i in range(len(sorted_nodes)) ] )

        full_weight = full_weight.flatten()
        row = row.flatten()
        col = col.flatten()

        return sp.csr_matrix( (full_weight, (row, col)), shape=(len(sorted_nodes), len(self.vocab)) )
    def _one_hot_(self, G, tf, sorted_nodes ):
        w = np.array(tf / np.sum(tf)).flatten()
        row = range(len(sorted_nodes))
        col = sorted_nodes
        return sp.csr_matrix( (w, (row, col)), shape=(len(sorted_nodes), len(self.vocab)) )

    def _build_features_(self, param):
        G, tf, tfidf, pg = param
        if len(G) == 0:
            G.add_node( self.vocab['<UNK>'] )
            return G, sp.csr_matrix(np.zeros(len(self.vocab)), shape=(1, len(self.vocab))) #sp.csr_matrix(np.random.normal(size=len(self.vocab)))
        
        nA = self._feature_definer_( G, tf, tfidf, pg, sorted( G.nodes ) )

        return G, nA
    
    def _feature_definer_( self, G, tf, tfidf, pg, sorted_nodes ):
        if self.feature_type == 'one_hot':
            return self._one_hot_( G, tf, sorted_nodes )
        if self.feature_type == 'full_weight':
            return self._full_weight_( tfidf, pg, sorted_nodes )
        if self.feature_type == 'prob':
            return self._prob_cooccur_( G, tf, sorted_nodes )
        
        if self.feature_type == 'full_weight_prob':
            full_weight = self._full_weight_( tfidf, pg, sorted_nodes )
            prob        = self._prob_cooccur_( G, tf, sorted_nodes )
            return full_weight.multiply(prob)
    
    
    def _build_graph_(self, doc):
        terms = list(filter( lambda x: x in self.vocab, doc))
        terms = list(map( lambda x: self.vocab[x], terms ))
        sorted_terms = sorted(list(set(terms)))
        
        tf = Counter(terms)
        tfidf = dict( [ (k, v*np.log2((self.N+1)/self.df[self.id2term[k]])) for (k,v) in tf.items() ] )

        cooccur_count = Counter()
        for i,idt in enumerate(terms):
            terms_to_add = terms[ max(i-self.w, 0):i ]
            terms_to_add = list(zip(terms_to_add, repeat(idt)))
            terms_to_add = list(map(sorted,terms_to_add))
            terms_to_add = list(map(tuple,terms_to_add))
            cooccur_count.update( terms_to_add )
        
        G = nx.Graph()
        G.add_nodes_from( [ (k,{'tfidf': tfidf[k], 'tf': tf[k] }) for k in set(terms) ] )
        w_edges = [ (s,t,w) for ((s,t),w) in cooccur_count.items() ]
        G.add_weighted_edges_from( w_edges, weight='freq' )
        
        
        tfidf = [ tfidf[term] for term in sorted_terms ]
        tf = [ tf[term] for term in sorted_terms ]
        #Add self-loops
        G.add_weighted_edges_from( zip(sorted_terms, sorted_terms, tf), weight='freq' )
        pg = nx.pagerank( G )
        pg = [ pg[term] for term in sorted_terms ]
        
        return G, tf, tfidf, pg

class ClassifierGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_heads=8, drop=.5, attn_drop=.5, device='cuda:0'):
        super(ClassifierGAT, self).__init__()

        self.encoder = nn.Linear(in_dim, hidden_dim).to(torch.device(device))
        
        self.layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, residual=True, num_heads=n_heads, activation=F.leaky_relu,
                    feat_drop=drop, attn_drop=attn_drop).to(torch.device(device)),
            GATConv(hidden_dim, hidden_dim, residual=True, num_heads=n_heads, activation=F.leaky_relu,
                    feat_drop=drop, attn_drop=attn_drop).to(torch.device(device))
        ])

        self.down_proj = [
            nn.Linear(n_heads*hidden_dim, hidden_dim).to(torch.device(device)),
            nn.Linear(n_heads*hidden_dim, hidden_dim).to(torch.device(device))
        ]
        
        self.lin = nn.Linear(hidden_dim + hidden_dim, 1).to(torch.device(device))
        self.pooling = GlobalAttentionPooling( self.lin ).to(torch.device(device))
        
        self.norm = nn.BatchNorm1d( hidden_dim + hidden_dim )
        self.drop = nn.Dropout(drop)
        
        self.classify = nn.Linear( hidden_dim + hidden_dim, n_classes).to(torch.device(device))

    def transform(self, G):
        h = G.ndata['f']
        he = self.encoder(h)
        h = he
        for l, conv in enumerate(self.layers):
            h = conv(G, h)
            h = h.view(h.shape[0], -1)
            # apply normlayer and scalling the dot-product attentions by the square-root
            h = self.down_proj[l]( h )
        # CONCAT he E hg
        hg = torch.cat((h,he), 1)
        hg = self.norm( hg )
        hg = self.drop( hg )
        return self.pooling(G, hg)
    def predict(self, G):
        hg = self.transform(G)
        pred = self.classify( hg )
        pred = torch.softmax(pred, 1)
        pred = torch.argmax(pred, 1).reshape(-1)
        return pred
    
    def forward(self, G):
        h = G.ndata['f']
        he = self.encoder(h)
        h = he
        for l, conv in enumerate(self.layers):
            h = conv(G, h)
            h = h.view(h.shape[0], -1)
            h = self.down_proj[l]( h )
        # CONCAT he AND h
        hg = torch.cat((h,he), 1)
        hg = self.norm( hg )
        hg = self.drop( hg )
        hg = self.pooling(G, hg)

        return self.classify( hg )

class FocalLoss(nn.Module):
    # https://github.com/mbsariyildiz/focal-loss.pytorch
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()