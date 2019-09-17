from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

import numpy as np
import io
from os import path
from glob import glob
from tqdm import tqdm

from multiprocessing import Pool,cpu_count

from sklearn.base import clone as clone_estimator
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics.pairwise import paired_distances, paired_cosine_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from collections import Counter

from random import shuffle
from itertools import repeat

from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
stemmer = SnowballStemmer("english")
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

def generate_lines(X):
    for i in range(X.shape[0]):
        yield (X[i].multiply(X[i].T))

def paired_di(params):
    X, models_ = params
    m = sp.lil_matrix((1, len(models_)*X.shape[1]))
    for i, ci_dom in enumerate(models_):
        m[0,i*X.shape[1]:(i+1)*X.shape[1]] = paired_cosine_distances(ci_dom, X)
    return m.tocsr()

def _paired_distances_(params):
    X, model = params
    return sp.csr_matrix(paired_cosine_distances(model, X))
    
def transform_class_repr(X, model):
    all_docs = sp.lil_matrix(X.shape)
    with Pool(processes=cpu_count()) as pool:
        lines = generate_lines(X)
        repeated_model = repeat(model)
        params = zip(lines, repeated_model)
        for i, m in tqdm(enumerate(pool.imap(_paired_distances_, params)), desc='Building class transformation', total=X.shape[0], smoothing=0.):
            all_docs[i,:] = m
    return all_docs

class BoWS(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=2, stop_words='english', alpha=0.1):
        self.min_df = min_df
        self.stop_words = stop_words
        self._cv = StemmedCountVectorizer(min_df=self.min_df, stop_words=self.stop_words)
        self._le = LabelEncoder()
        self.alpha = alpha
        self._fitted_ = False
        self.models_ = []
    def __del__(self):
        del self.models_[:]
    def fit(self, X_texts, y=None):
        if y is None:
            raise TypeError("y can't be None")
            
        a = list(zip(X_texts, y))
        shuffle(a)
        X_texts, scores = list(zip(*a))
        X_texts = list(X_texts)
        y = list(scores)
            
        X = self._build_binary_cooccur_matrix_(X_texts)
        y = self._normalize_y_(y)
        
        self._build_auxiliar_features(X, y)
        self._build_class_models_()
        
        del self.Ntc_
        del self.Nt_
        del self.Pt_
        
        self._fitted_ = True
        
        return self
    def transform(self, X_texts):
        if not self._fitted_:
            raise TypeError("The model did'nt fit yet!")
            
        X = self._cv.transform(X_texts)
        X_classes = {}
        for c in range(self.C_):
            X_classes[self._le.inverse_transform([c])[0]] = transform_class_repr(X, self.models_[c].copy())
        return X_classes
    def _build_binary_cooccur_matrix_(self, texts):
        X_TF = self._cv.fit_transform(texts).tocsr()    
        X = sp.csr_matrix( ( np.ones(len(X_TF.data)), X_TF.nonzero() ), shape=X_TF.shape )
        del X_TF
        return X
    def _normalize_y_(self, y):
        return self._le.fit_transform(y)
    def _build_auxiliar_features(self, X, y):
        # número de documentos
        self.N_ = X.shape[0]

        # tamanho do vocabulário
        self.V_ = X.shape[1] 

        # Número de classes
        self.C_ = max(y)+1

        # Número de cada co-ocrrência por classe
        self.Ntc_ = [ sp.lil_matrix( (self.V_,self.V_) ) for _ in range(max(y)+1) ]
        for i, doc_matrix in tqdm(enumerate(generate_lines(X)), total=self.N_, desc='Building class representations'):
            self.Ntc_[ y[i] ] = (self.Ntc_[ y[i] ] + doc_matrix)
        ### Remove diagonal principal
        for i in range(len(self.Ntc_)):
            self.Ntc_[i].setdiag(0)
            self.Ntc_[i].eliminate_zeros()
        
        # frequencia de cada co-ocorrência por classe
        self.Nt_ = np.sum(self.Ntc_)

        # priori de cada termo P(t)
        self.Pt_ = self.Nt_/self.N_
        self.Pt_.eliminate_zeros()
    def _build_class_models_(self):
        self.models_ = []
        for i in tqdm(range(self.C_), total=self.C_, desc='Building Models'):
            # Probabilidade P(t,c)
            data = np.array(self.Ntc_[i][ self.Nt_.nonzero() ] / self.Nt_[ self.Nt_.nonzero() ])[0]
            Ptc = sp.csr_matrix( (data, self.Nt_.nonzero()), shape=self.Nt_.shape )

            # Jenilek-Mercer smoothing
            norm_Ptc = (1.-self.alpha)*Ptc + self.alpha*self.Pt_

            # P*sqrt(n)
            data1 = np.multiply(norm_Ptc[norm_Ptc.nonzero()], np.sqrt(self.Ntc_[i][norm_Ptc.nonzero()]))
            # 2*sqrt( p(1-p) )
            data2 = 2.*np.sqrt(np.multiply(norm_Ptc[ norm_Ptc.nonzero() ], 1.-norm_Ptc[ norm_Ptc.nonzero() ]))

            CI_dominance_smooth = sp.csr_matrix( (np.array(data1/data2)[0], norm_Ptc.nonzero()), shape=norm_Ptc.shape )

            del data
            del data1
            del data2
            
            max_prob = (1.-self.alpha)*Ptc.data.max() + self.alpha*self.Pt_.data.max()
            max_size_ic = (max_prob*np.sqrt(self.Ntc_[i].data.max())) / (2.*np.sqrt( max_prob*(1.-max_prob) ))

            CI_dominance_smoooth_norm = CI_dominance_smooth / max_size_ic
            CI_dominance_smoooth_norm.eliminate_zeros()

            self.models_.append(CI_dominance_smoooth_norm)
            del CI_dominance_smooth
            del norm_Ptc
            del Ptc


class OneVsAllGridClassifier(object):
    def __init__(self, weak_params, weak_classifier, meta_params, meta_classifier,
                 cv=5, n_jobs=12, scoring='f1_micro', iid=True):
        self.cv = cv
        self.n_jobs = n_jobs
        
        self.weak_clf = clone_estimator(weak_classifier)
        self.weak_params = weak_params
        self.weak_gs = GridSearchCV(self.weak_clf, self.weak_params, cv=self.cv, n_jobs=self.n_jobs, scoring=scoring, iid=iid)
        
        self.meta_clf = clone_estimator(meta_classifier)
        self.meta_params = meta_params
        self.meta_gs = GridSearchCV(self.meta_clf, self.meta_params, cv=self.cv, n_jobs=self.n_jobs, scoring=scoring, iid=iid)
        
        self.clf_by_class = {}
        self._fitted_ = False
        
    def fit(self, X_multiclass, y):
        self.classes_ = sorted( list(X_multiclass.keys()) )
        self.classes_mapper_ = { k:i for (i,k) in enumerate(self.classes_) }
        
        X_probs = []
        for c in self.classes_:
            X_class = X_multiclass[c]
            
            y_transformed = self.transform_y(y,c)
            weak_gs_atual = clone_estimator(self.weak_gs)
            
            weak_gs_atual.fit(X_class, y_transformed)
            
            clf = clone_estimator(self.weak_clf).set_params(**weak_gs_atual.best_params_)

            cccv = CalibratedClassifierCV(clf, cv=self.cv)
            
            self.clf_by_class[c] = cccv.fit(X_class, y_transformed)
            X_probs.append( cccv.predict_proba( X_class ) )
            
        X_probs = np.matrix(X_probs).T
        self.meta_gs.fit(X_probs, y)
        self.meta_clf.set_params(**self.meta_gs.best_params_)
        self.meta_clf.fit(X_probs, y)
        
        self._fitted_ = True
        
        return self
    
    def predict(self, X_multiclass):
        if not self._fitted_ :
            raise Exception("Model did'nt fit.")
        X_probs = self.predict_proba_weak(X_multiclass)
        y_pred = self.meta_clf.predict(X_probs)
        return y_pred
    
    def predict_proba_weak(self, X_multiclass):
        if not self._fitted_ :
            raise Exception("Model did'nt fit.")
        X_probs = []
        for c in self.classes_:
            X_class = X_multiclass[c]
            X_probs.append( self.clf_by_class[c].predict_proba( X_class ) )
        return np.matrix(X_probs).T
    def transform_y(self, y, c):
        return np.array([ a_ == c for a_ in y ], dtype=int)
    def fit_predict(self, X_multiclass, y):
        return self.fit(X_multiclass, y).predict(X_multiclass)