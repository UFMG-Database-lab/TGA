from BoTG_line_graph import BoTG
from BoTG.DataRepresentation import Document
from glob import glob
import argparse
from os import path
import os

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import dump_svmlight_file
import numpy as np

import io
import gzip

def dump_svmlight_file_gz(X,y,filename):
    with gzip.open(filename, 'w') as filout:
        dump_svmlight_file(X, y, filout, zero_based=False)
def readfile(filename):
    with io.open(filename, 'rt', newline='\n', encoding='utf8', errors='ignore') as filein:
        return filein.readlines()

def load_splits_ids(folddir):
    splits = []
    with open(folddir, encoding='utf8', errors='ignore') as fileout:
        for line in fileout.readlines():
            train_index, test_index = line.split(';')
            train_index = list(map(int, train_index.split()))
            test_index = list(map(int, test_index.split()))
            splits.append( (train_index, test_index) )
    return splits 
def get_array(X, idxs):
    return [ X[idx] for idx in idxs ]
parser = argparse.ArgumentParser()

required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-d','--datasetdir', type=str, nargs='+', help='', required=True)

# w=2, min_df=2, metric='cosine', pooling='mean', assignment='hard'
parser.add_argument('--silence', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-f','--nfolds', type=int, nargs='?', help='', default=5)

parser.add_argument('-w','--window', type=int, nargs='+', help='', default=[2])
parser.add_argument('-e','--eps', type=float, nargs='+', help='', default=[None])

parser.add_argument('-tt', '--train_test', action="store_true", help='[Optional] (default=False) build only train_test fold (or, zero-based fold).')

args = parser.parse_args()
# Sample:

# nohup python3 execute_dataset.py -d datasetpath -w 1 2 3 -df 2 3 4 -p mean max sum -a unorm hard -m cosine l2 precomputed -dir in out both 2>&1 > ../LOG_dataset.txt &

if __name__ == '__main__':
    for d in tqdm(args.datasetdir, total=len(args.datasetdir), desc="Running datasets", position=0):
        dname = path.basename(path.dirname(d))
        doc_texts = readfile(path.join(d,'texts.txt'))
        y = list(map(int, readfile(path.join(d,'score.txt'))))
        splits_folds = load_splits_ids(path.join(d, 'representations','split_%d.csv' % args.nfolds))

        if args.train_test:
            splits_folds = [splits_folds[0]]

        for w in args.window:
            for eps in args.eps:
                for f, (train_index, test_index) in tqdm(enumerate(splits_folds), position=1, desc="Running Folds on %s" % dname, total=len(splits_folds)):
                    botg = BoTG(w=w, eps=eps)
                    docs_train, y_train = get_array(doc_texts, train_index), get_array(y,train_index)
                    docs_test, y_test = get_array(doc_texts,test_index), get_array(y,test_index)
                    botg.fit(docs_train, verbose=not args.silence)

                    name_file_config = f"BoTG_TF-Degree_e{eps}_w{w}"
                    output_path =  path.join(d,'representations', f'{args.nfolds}-folds', name_file_config)
                    if not path.exists( output_path ):
                        os.makedirs(output_path)

                    filename_train = "train%d.gz" % f
                    output_file = path.join(output_path,filename_train)
                    X = botg.transform(docs_train, verbose=not args.silence)
                    dump_svmlight_file_gz(X,y_train, output_file)

                    filename_test = "test%d.gz" % f
                    output_file = path.join(output_path,filename_test)
                    X = botg.transform(docs_test, verbose=not args.silence)
                    dump_svmlight_file_gz(X,y_test, output_file)
