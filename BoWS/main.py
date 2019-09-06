import io

from sklearn.metrics import f1_score

import argparse
from os import path
import os
import gzip
from sklearn.datasets import dump_svmlight_file

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from base import BoWS, OneVsAllGridClassifier

def read_texts(filename):
    with io.open(filename, newline='\n') as filin:
        return filin.readlines()

def get_array(X, idxs):
    return [ X[idx] for idx in idxs ]
def read_dataset(pathname):
    texts = read_texts(path.join(pathname, 'texts.txt'))
    scores = read_texts(path.join(pathname, 'score.txt'))
    scores = list(map(int, scores))
    return texts,scores
def dump_svmlight_file_gz(X,y,filename):
    with gzip.open(filename, 'w') as filout:
        dump_svmlight_file(X, y, filout, zero_based=False)
def load_splits_ids(folddir):
    splits = []
    with open(folddir, encoding='utf8', errors='ignore') as fileout:
        for line in fileout.readlines():
            train_index, test_index = line.split(';')
            train_index = list(map(int, train_index.split()))
            test_index = list(map(int, test_index.split()))
            splits.append( (train_index, test_index) )
    return splits 

parser = argparse.ArgumentParser()

required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-d','--datasetdir', type=str, nargs='+', help='', required=True)

parser.add_argument('-f','--nfolds', type=int, nargs='?', help='', default=5)
parser.add_argument('-df','--min_df', type=int, nargs='?', help='', default=2)

args = parser.parse_args()

for datasetdir in args.datasetdir:
    dname = path.basename(path.dirname(datasetdir))
    train_test_splits = load_splits_ids(path.join(datasetdir, 'representations', 'split_%d.csv' % args.nfolds ))
    texts,scores = read_dataset(datasetdir)
    for f, (train_ids, test_ids) in enumerate(train_test_splits):
        print(dname, f, len(train_ids), len(test_ids))
        X_train = get_array(texts, train_ids)
        y_train = get_array(scores, train_ids)

        bows = BoWS(min_df=args.min_df)
        X_train_transformed = bows.fit_transform(X_train, y_train)

        weak_clf = LogisticRegression(random_state=42)
        weak_params = {'penalty': ['l1', 'l2'], 'class_weight': ['balanced', None], 'solver': ['liblinear'], 'C': [1, 10, 0.1, 0.01]}
        #weak_params = {'penalty': ['l2'], 'class_weight': [None], 'solver': ['liblinear'], 'C': [0.01]}
        meta_clf = DecisionTreeClassifier()
        meta_params = { 'criterion': [ "gini", "entropy" ] }
        
        oal = OneVsAllGridClassifier( weak_params, weak_clf, meta_params, meta_clf )
        
        y_pred = oal.fit_predict(X_train_transformed, y_train)
        f1_micr_train = f1_score(y_train, y_pred, average='micro')
        f1_macr_train = f1_score(y_train, y_pred, average='macro')
        print( "[%s] - [TRAIN] F1_micro: %.3f F1_macro: %.3f" % (dname, f1_micr_train, f1_macr_train) )

        X_test = get_array(texts, test_ids)
        y_test = get_array(scores, test_ids)
        
        X_test_transformed = bows.transform(X_test)
        y_pred = oal.predict(X_test_transformed)
        f1_micr_test = f1_score(y_test, y_pred, average='micro')
        f1_macr_test = f1_score(y_test, y_pred, average='macro')
        print( "[%s] - [TEST] F1_micro: %.3f F1_macro: %.3f" % (dname, f1_micr_test, f1_macr_test) )



