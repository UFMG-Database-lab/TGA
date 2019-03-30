from BoTG.BoTG import BoTG
from BoTG.DataRepresentation import Document
from glob import glob
import argparse
from os import path
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import dump_svmlight_file
import numpy as np

import io

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
parser.add_argument('-df','--min_df', type=int, nargs='+', help='', default=[2])
parser.add_argument('-e','--eps', type=float, nargs='+', help='', default=[0.1])

parser.add_argument('-p','--pooling', type=str, nargs='+', help='', default=['mean'], choices=['mean', 'max', 'sum'])
parser.add_argument('-a','--assignment', type=str, nargs='+', help='', default=['hard'], choices=['hard', 'unorm'])
parser.add_argument('-m','--metric', type=str, nargs='+', help='', default=['cosine'])
parser.add_argument('-dir','--direction', type=str, nargs='+', help='', default=['both'], choices=['in', 'out', 'both'])
parser.add_argument('-tt', '--train_test', action="store_true", help='[Optional] (default=False) build only train_test fold (or, zero-based fold).')

parser.add_argument('-o','--output', type=str, nargs='?', help='', default='output')

args = parser.parse_args()

if not path.exists(args.output):
    try:
        os.makedirs(args.output)
    except:
        print("Couldn't create the %s directory" % args.output)


# Sample:

# nohup python3 execute_dataset.py -d datasetpath -w 1 2 3 -df 2 3 4 -p mean max sum -a unorm hard -m cosine l2 precomputed -dir in out both 2>&1 > ../LOG_dataset.txt &

if __name__ == '__main__':
    for d in args.datasetdir:
        dname = path.basename(path.dirname(d))
        doc_texts = readfile(path.join(d,'texts.txt'))
        y = list(map(int, readfile(path.join(d,'score.txt'))))
        splits_folds = load_splits_ids(path.join(d, 'folds','split_%d.csv' % args.nfolds))

        for w in args.window:
            docs = np.array(Document.build_docs(doc_texts, w=w, verbose=not args.silence))
            for df in args.min_df:
                for m in args.metric:
                    for direction in args.direction: 
                        for eps in args.eps:
                            for f, (train_index, test_index) in enumerate(splits_folds):
                                print("fold%d_%s_e%.2f_w%d_df%d_m-%s_dir-%s" % (f, dname, eps, w, df, m, direction))
                                botg = BoTG(metric=m, min_df=df, direction=direction, quantile=eps)
                                #botg = BoTG(metric=m, min_df=df, memory_strategy=args.memory_strategy)
                                docs_train, y_train = get_array(docs, train_index), get_array(y,train_index)
                                print("fitting")
                                botg.fit(docs_train, verbose=not args.silence)
                                print("transforming %s" % botg)
                                for p in args.pooling:
                                    for a in args.assignment:
                                        filename_train = "train%d_%s_e%.2f_w%d_df%d_m-%s_p-%s_a-%s" % (f,dname,eps,w,df,m,p,a)
                                        print("  %s" % filename_train)
                                        X = botg.transform(docs_train, pooling=p, assignment=a, verbose=not args.silence)
                                        dump_svmlight_file(X,y_train, path.join(args.output,filename_train))

                                        docs_test, y_test = get_array(docs,test_index), get_array(y,test_index)
                                        filename_test = "test%d_%s_e%.2f_w%d_df%d_m-%s_p-%s_a-%s" % (f,dname,eps,w,df,m,p,a)
                                        print("  %s" % filename_test)
                                        X = botg.transform(docs_test, pooling=p, assignment=a, verbose=not args.silence)
                                        dump_svmlight_file(X,y_test, path.join(args.output,filename_test))
                                botg.close()
                                if args.train_test:
                                    break
