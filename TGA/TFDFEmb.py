import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from tokenizer import Tokenizer

from tqdm import tqdm
import numpy as np

import copy

from multiprocessing import cpu_count

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return torch.cos(sim_mt)

class AttentionTFIDF(nn.Module):
    def __init__(self, vocab_size, hiddens, nclass, maxF=20, drop=.5):
        super(AttentionTFIDF, self).__init__()
        self.hiddens        = hiddens
        self.maxF           = maxF
        self.value_emb      = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.query_emb      = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.key_emb        = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.TF_emb         = nn.Embedding(maxF, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.DF_emb         = nn.Embedding(maxF, hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.fc             = nn.Linear(hiddens, nclass)
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
        h_query = torch.tanh( h_query )
        h_query = F.dropout( h_query, p=self.drop_, training=self.training )
        
        h_key = self.key_emb( doc_tids )
        h_key = h_key + h_TFs + h_DFs
        h_key = torch.tanh( h_key )
        h_key = F.dropout( h_key, p=self.drop_, training=self.training )
        
        co_weights  = sim_matrix( h_key, h_query )
        
        co_weights[pad_mask.logical_not()] = 0. # Set the 3D-pad mask values to
        co_weights = torch.relu(co_weights)
        
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
        nn.init.xavier_normal_(self.TF_emb.weight.data)
        nn.init.xavier_normal_(self.DF_emb.weight.data)
        nn.init.xavier_normal_(self.query_emb.weight.data)
        nn.init.xavier_normal_(self.key_emb.weight.data)
        nn.init.xavier_normal_(self.value_emb.weight.data)
        nn.init.xavier_normal_(self.fc.weight.data)

class AttentionTFIDFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hiddens=300, mindf=2, lan='english', stopwords='nltk', k=512,
                 max_drop=.85,
                 batch_size = 64, lr=5e-3, weight_decay=5e-3,
                 nepochs = 1000, patience=10, factor=.95,
                 n_jobs=cpu_count(), _device=torch.device('cuda:0'), _verbose=False):
        super(AttentionTFIDFClassifier, self).__init__()

        self._model         = None
        self._tokenizer     = None
        self.nepochs        = int(nepochs)
        self.hiddens        = int(hiddens)
        self.mindf          = int(mindf)
        self.lan            = lan
        self.stopwords      = stopwords
        self.k              = int(k)
        self.max_drop       = max_drop
        self._verbose       = _verbose
        self._device        = _device

        self.n_jobs         = int(n_jobs)
        
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.patience       = int(patience)
        self.factor         = factor
        self.batch_size     = int(batch_size)
        
        def collate_train(param):
            X, y = zip(*param)
            y = self._tokenizer.le.transform(y)
            doc_tids, TFs, DFs = self._tokenizer.transform(X, verbose=False)
            
            doc_tids = pad_sequence(list(map(torch.LongTensor, doc_tids)), batch_first=True, padding_value=0)

            TFs = pad_sequence(list(map(torch.tensor, TFs)), batch_first=True, padding_value=0)
            TFs = torch.LongTensor(torch.log2(TFs+1).round().long())

            DFs = pad_sequence(list(map(torch.tensor, DFs)), batch_first=True, padding_value=0)
            DFs = torch.LongTensor(torch.log2(DFs+1).round().long())

            return doc_tids, TFs, DFs, torch.LongTensor(y)
        def collate_predict(param):
            X = zip(*param)
            doc_tids, TFs, DFs = self._tokenizer.transform(X, verbose=False)
            
            doc_tids = pad_sequence(list(map(torch.LongTensor, doc_tids)), batch_first=True, padding_value=0)

            TFs = pad_sequence(list(map(torch.tensor, TFs)), batch_first=True, padding_value=0)
            TFs = torch.LongTensor(torch.log2(TFs+1).round().long())

            DFs = pad_sequence(list(map(torch.tensor, DFs)), batch_first=True, padding_value=0)
            DFs = torch.LongTensor(torch.log2(DFs+1).round().long())

            return doc_tids, TFs, DFs
        
        self.collate_train = collate_train
        self.collate_predict = collate_predict
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            pass
        self._tokenizer = Tokenizer(mindf=self.mindf, lan=self.lan, stopwordsSet=self.stopwords,
                                   model='sample', k=self.k, verbose=self._verbose)
        self._tokenizer.fit(X_train, y_train)

        self.maxF = int(round(np.log2(self._tokenizer.maxF+1))) 
        
        self._model     = AttentionTFIDF( vocab_size=self._tokenizer.vocab_size, hiddens=self.hiddens,
                                        nclass=self._tokenizer.n_class, maxF=self.maxF, drop=self.max_drop ).to(self._device)
        
        
        
        optimizer = optim.AdamW( self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_func_cel = nn.CrossEntropyLoss().to( self._device )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.factor,
                                                       patience=3, verbose=self._verbose)

        
        best = 99999.
        best_acc = 0.
        counter = 1
        dl_val = DataLoader(list(zip(X_val, y_val)), batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.collate_train, num_workers=self.n_jobs)

        for e in tqdm(range(self.nepochs), total=self.nepochs, disable=not self._verbose):
            dl_train = DataLoader(list(zip(X_train, y_train)), batch_size=self.batch_size,
                                    shuffle=True, collate_fn=self.collate_train, num_workers=self.n_jobs)
            loss_train  = 0.
            with tqdm(total=len(y_train)+len(y_val), smoothing=0., desc=f"ACC_val: {best_acc:.2} Epoch {e+1}", disable=not self._verbose) as pbar:
                total = 0
                correct  = 0
                self._model.train()
                self._tokenizer.model = 'sample'
                for i, (doc_tids, TFs, DFs, y) in enumerate(dl_train):

                    doc_tids = doc_tids.to(self._device)
                    TFs = TFs.to(self._device)
                    DFs = DFs.to(self._device)
                    y = y.to(self._device)

                    pred_docs,_,_ = self._model( doc_tids, TFs, DFs )
                    pred_docs     = torch.softmax(pred_docs, dim=1)
                    loss          = loss_func_cel(pred_docs, y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    loss_train += loss.item()
                    total      += len(y)
                    y_pred      = pred_docs.argmax(axis=1)
                    correct    += (y_pred == y).sum().item()
                    self._model.drop_ =  (correct/total)*self.max_drop
                    
                    pbar.update( len(y) )
                    del doc_tids, TFs
                    del DFs, y, pred_docs
                    del loss, y_pred
                loss_train = loss_train/(i+1)
                total = 0
                correct  = 0
                self._model.eval()
                self._tokenizer.model = 'topk'
                with torch.no_grad():
                    loss_val = 0.
                    for i, (doc_tids, TFs, DFs, y) in enumerate(dl_val):
                        doc_tids = doc_tids.to(self._device)
                        TFs = TFs.to(self._device)
                        DFs = DFs.to(self._device)
                        y = y.to(self._device)

                        pred_docs,_,_ = self._model( doc_tids, TFs, DFs )
                        pred_docs     = torch.softmax(pred_docs, dim=1)
                        loss          = loss_func_cel(pred_docs, y)

                        loss_val   += loss.item()
                        total      += len(y)
                        y_pred      = pred_docs.argmax(axis=1)
                        correct    += (y_pred == y).sum().item()
                        pbar.update( len(y) )
                        loss_val
                        del doc_tids, TFs, DFs, y
                        del pred_docs, loss
                    loss_val   = (loss_val/(i+1))
                    scheduler.step(loss_val)

                    if best-loss_val > 0.0001 :
                        best = loss_val
                        counter = 1
                        best_acc = correct/total
                        best_model = copy.deepcopy(self._model).to('cpu')
                    elif counter > self.patience:
                        break
                    else:
                        counter += 1
        
        self._model = best_model.to( self._device )

        self._loss = best
        self._acc  = best_acc

        return self

    def predict(self, X):
        if self._model is None or self._tokenizer is None:
            raise Exception("Not implemented yet!")
        self._model.eval()
        self._tokenizer.model = 'topk'
        dataloader = DataLoader(X, batch_size=self.batch_size,
                        shuffle=False, collate_fn=self.collate_predict, num_workers=self.n_jobs)
        result = []
        with torch.no_grad():
            loss_val = 0.
            for i, (doc_tids, TFs, DFs) in enumerate(dataloader):
                doc_tids = doc_tids.to(self._device)
                TFs = TFs.to(self._device)
                DFs = DFs.to(self._device)

                pred_docs,_,_ = self._model( doc_tids, TFs, DFs )
                pred_docs     = torch.softmax(pred_docs, dim=1)
                result.extend( list(pred_docs) )
        return self._tokenizer.le.inverse_transform(np.array(result))
