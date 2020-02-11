from utils import FocalLoss, ClassifierGAT, GraphsizePretrained
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from torch.utils.data import DataLoader
import torch.optim as optim
from os import path
from tqdm import tqdm
import torch.nn as nn
import torch
import dgl

from datetime import datetime


class TGA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
        Textual Graph Attention (TGA)
    """
    instances = 0
    def __init__(self, pretrained_vec, w=5, hidden_dim=300,
                 drop=.5, attn_drop=.3, n_heads=4, n_epochs=1000,
                 patience=25, train_batch_size=16,
                 transform_batch_size=256, lr=1e-3,
                 weight_decay=1e-3, optim_name='adam',
                 loss_name='cross_entropy', device='cuda:0',
                 verbose=False):
        self._pretrained_vec = pretrained_vec
        self.pretrained_vec = path.basename(self._pretrained_vec)
        self.w = w
        self.hidden_dim = hidden_dim
        self.drop = drop
        self.attn_drop = attn_drop
        self.n_heads = n_heads
        self.n_epochs = n_epochs
        self.patience = patience
        self.train_batch_size = train_batch_size
        self.transform_batch_size = transform_batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optim_name = optim_name
        self.loss_name = loss_name 
        self.device = device
        self.verbose = verbose
        
        self.graph_builder = GraphsizePretrained(w=self.w,
                                                 pretrained_vec=self._pretrained_vec,
                                                 verbose=verbose)
        self.in_dim = self.graph_builder.ndim
        
        if self.loss_name.lower() == 'focal':
            self.loss_func = FocalLoss().to(torch.device(device))  
        elif self.loss_name.lower() == 'cross_entropy':
            self.loss_func = nn.CrossEntropyLoss().to(torch.device(device))
        TGA.instances += 1
        self.path_to_save = f'best_param_{TGA.instances}_{datetime.now().isoformat()}.pth'
    def __del__(self):
        TGA.instances -= 1
        
    def collate_train(self, samples):
        Gs_Fs, labels = map(list, zip(*samples))
        graphs = []
        for g, f in Gs_Fs:
            g_dgl = dgl.DGLGraph()
            g_dgl.from_networkx(g)
            g_dgl.ndata['f'] = torch.FloatTensor(f).to(torch.device(self.device ))
            g_dgl.to(torch.device(self.device ))
            graphs.append(g_dgl)
        batched_graph = dgl.batch(graphs)
        batched_graph.to(torch.device(self.device ))
        labels = torch.tensor(labels).to(torch.device(self.device ))
        return batched_graph, labels
    def collate_test(self, samples):
        Gs_Fs = samples
        graphs = []
        for g, f in Gs_Fs:
            g_dgl = dgl.DGLGraph()
            g_dgl.from_networkx(g)
            g_dgl.ndata['f'] = torch.FloatTensor(f).to(torch.device(self.device))
            g_dgl.to(torch.device(self.device))
            graphs.append(g_dgl)
        batched_graph = dgl.batch(graphs)
        batched_graph.to(torch.device(self.device))
        return batched_graph   
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        
        self.nclass= len(set(y_train))
        Gs_train = self.graph_builder.fit_transform( X_train )
        Gs_val   = self.graph_builder.transform( X_val )
        
        self.model = ClassifierGAT(self.in_dim, self.hidden_dim, self.nclass,
                                   n_heads=self.n_heads, attn_drop=self.attn_drop,
                                   drop=self.drop).to(torch.device(self.device))
        
        data_loader_val = DataLoader(list(zip(Gs_val, y_val)), batch_size=self.train_batch_size,
                                     shuffle=False, collate_fn=self.collate_train)
        
        if self.optim_name.lower() == 'adam':
            self.optimizer = optim.Adam( self.model.parameters(),
                                        lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim_name.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop( self.model.parameters(),
                                           lr=self.lr, weight_decay=self.weight_decay )
            
        best_score = None
        n_iters = 0
        
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            torch.cuda.synchronize()
            
        self.model.train()

        for epoch in range(self.n_epochs):
            data_loader = DataLoader(list(zip(Gs_train, y_train)),
                                     batch_size=self.train_batch_size,
                                     shuffle=True, collate_fn=self.collate_train)
            epoch_loss = 0
            with tqdm(total=len(data_loader.dataset), smoothing=0., disable=not self.verbose) as pbar:
                total = 0
                correct = 0
                self.model.train()
                for i, (bg, label) in enumerate(data_loader):
                    outputs = self.model(bg)
                    probs_Y = torch.softmax(outputs, 1)
                    sampled_Y = torch.argmax(probs_Y, 1).reshape(-1)

                    # Train eval phase
                    total += label.size(0)
                    correct += (sampled_Y == label).sum().item()

                    # NN backprop phase
                    loss = self.loss_func(outputs, label)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.detach().item()

                    del loss, outputs, bg, probs_Y, sampled_Y
                    pbar.update( len(label) )
                    pbar.set_description_str(f"iter {epoch}, train acc {(correct/total):.3f} train loss {(epoch_loss/(epoch + 1)):.3f}")

                score_train = correct/total
            with tqdm(total=len(data_loader_val.dataset), smoothing=0., disable=not self.verbose) as pbar:
                self.model.eval()
                total = 0
                correct = 0
                for bg, label in data_loader_val:
                    with torch.no_grad():
                        outputs = self.model(bg)

                    probs_Y = torch.softmax(outputs, 1)
                    sampled_Y = torch.argmax(probs_Y, 1).reshape(-1)

                    # Validation eval phase
                    total += label.size(0)
                    correct += (sampled_Y == label).sum().item()

                    del probs_Y, outputs, bg, sampled_Y
                    score_val = correct/total

                    pbar.set_description_str(f'iter {epoch}, val   acc {score_val:.3f} ( over: {(score_val/score_train):.3} )')
                    pbar.update( label.size(0) )
                    
                pbar.set_description_str(f'iter {epoch}, val   acc {score_val:.3f} ( over: {(score_val/score_train):.3}/{n_iters} )')
                score = correct/total
                if best_score is None or score > best_score:
                    torch.save(self.model, self.path_to_save)
                    best_score = score
                    n_iters = 0
                else:
                    n_iters += 1
                    if n_iters >= self.patience:
                        pbar.set_description_str(f'iter {epoch}, val   acc {score_val:.3f} ( over: {(score_val/score_train):.3}/{n_iters} )')
                        break
        self.model = torch.load(self.path_to_save)
        return self
    def transform(self, X):
        pass
    def predict(self, X):
        pass