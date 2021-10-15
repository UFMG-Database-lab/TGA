from glob import glob
from collections import namedtuple
from os import path, remove
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import io
import numpy as np

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