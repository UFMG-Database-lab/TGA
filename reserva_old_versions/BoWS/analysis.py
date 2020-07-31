from base import BoWS, OneVsAllGridClassifier
import io

from sklearn.metrics import f1_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from imblearn.under_sampling import NearMiss
from sklearn.calibration import CalibratedClassifierCV

import argparse
import json
from glob import glob

from os import path
import os

from collections import Counter
import numpy as np

import gc

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
parser.add_argument('-u','--undersample', type=int, nargs='?', help='', default=2)
parser.add_argument('-o','--output', type=str, nargs='?', help='', default='tmp_output')

args = parser.parse_args()
if args.undersample == 0:
    undersampler = None
else:
    undersampler = NearMiss(version=args.undersample)

for datasetdir in args.datasetdir:
    texts,scores = read_dataset(datasetdir)
    train_test_splits = load_splits_ids(path.join(datasetdir, 'representations', 'split_%d.csv' % args.nfolds ))
    dname = path.basename(path.dirname(path.join(datasetdir,'')))
    to_save = { 'nfolds': args.nfolds, 'dname': dname }
    print(dname)
    folds = [ ]
    for f, (train_ids, test_ids) in enumerate(train_test_splits):
        fold_output = { 'id': f }
        X_train = get_array(texts, train_ids)
        y_train = get_array(scores, train_ids)
        fold_output['y_train'] = y_train

        bows = BoWS(min_df=2)
        X_train_transformed = bows.fit_transform(X_train, y_train)

        #weak_clf = LogisticRegression(random_state=42, max_iter=1000)
        #weak_params = {'penalty': ['l1', 'l2'], 'class_weight': ['balanced', None], 'solver': ['liblinear'], 'C': [1, 10, 0.1, 0.01]}
        
        weak_clf = LinearSVC(random_state=42, max_iter=100000)
        weak_params = {'C': [1, 10, 0.1, 0.01]}

        meta_clf = DecisionTreeClassifier()
        meta_params = { 'criterion': [ "gini", "entropy" ], 'max_depth': [None, 2, 4, 6], 'min_samples_split': [2,4,6], 'min_samples_leaf': [1, 2, 4, 6] }

        oal = OneVsAllGridClassifier( weak_params, weak_clf, meta_params, meta_clf )

        y_pred = oal.fit_predict(X_train_transformed, y_train)
        fold_output['y_train_pred'] = y_pred


        # ERRO: O GET PARAMS TA PEGANDO OS PARAMETROS 
        fold_output['weak_classifier'] = {}
        for c, weak_class in oal.clf_by_class.items():
            fold_output['weak_classifier'][str(c)] = {}
            wc = weak_class
            if type(weak_class) is CalibratedClassifierCV:
                wc = weak_class.base_estimator
            fold_output['weak_classifier'][str(c)]['classifier'] = str(type(wc))
            fold_output['weak_classifier'][str(c)]['params'] = wc.get_params()

        X_test = get_array(texts, test_ids)
        X_test_transformed = bows.transform(X_test)
        y_pred = oal.predict(X_test_transformed)
        fold_output['y_test_pred'] = y_pred

        y_test = get_array(scores, test_ids)
        fold_output['y_test'] = y_test

        fold_output['metaclassifier'] = { 'classifier': str(type(oal.meta_clf)), 'params': oal.meta_clf.get_params() }
        f1_micr = f1_score(y_pred, y_test, average='micro')
        f1_macr = f1_score(y_pred, y_test, average='macro')
        print("Fold %d (F1_mi: %.3f and F1_ma: %.3f)" % (f, f1_micr, f1_macr))
        fold_output['weak_class_result'] = {  }
        for c in X_test_transformed:
            fold_output['weak_class_result'][str(c)] = {}

            y_pred = oal.clf_by_class[c].predict(X_test_transformed[c])
            y_test_transformed = oal.transform_y(y_test, c)

            fold_output['weak_class_result'][str(c)]['y_test_transformed'] = y_test_transformed
            fold_output['weak_class_result'][str(c)]['y_pred'] = y_pred

            f1_micr = f1_score(y_pred, y_test_transformed, average='micro')
            f1_macr = f1_score(y_pred, y_test_transformed, average='macro')
            print("\t%d F1_mi: %.3f and F1_ma: %.3f" % ( c, f1_micr, f1_macr) )

        gc.collect()
        folds.append(fold_output)
    to_save['folds'] = folds
    filepath = path.join(args.output, dname + '.json')
    with open(filepath, 'w') as fp:
        json.dump(to_save, fp, cls=NumpyEncoder)