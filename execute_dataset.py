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

def load_labels(path_dataset):
    with open(path.join(path_dataset, 'doc.labeled')) as filin:
        data = [ line.split() for line in filin.readlines() ]
    return np.array(sorted(data, key=lambda x: x[0]))
    


parser = argparse.ArgumentParser()

required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-d','--datasetdir', type=str, nargs='+', help='', required=True)

# w=2, min_df=2, metric='cosine', pooling='mean', assignment='hard'
parser.add_argument('--silence', default=False, type=lambda x: (str(x).lower() == 'true'))


parser.add_argument('-f','--nfolds', type=int, nargs='?', help='', default=10)

parser.add_argument('-w','--window', type=int, nargs='+', help='', default=[2])
parser.add_argument('-df','--min_df', type=int, nargs='+', help='', default=[2])

parser.add_argument('-p','--pooling', type=str, nargs='+', help='', default=['mean'], choices=['mean', 'max', 'sum'])
parser.add_argument('-a','--assignment', type=str, nargs='+', help='', default=['hard'], choices=['hard', 'soft', 'unorm'])
parser.add_argument('-m','--metric', type=str, nargs='+', help='', default=['cosine'])

parser.add_argument('-o','--output', type=str, nargs='?', help='', default='output')

args = parser.parse_args()

if not path.exists(args.output):
    try:
        os.makedirs(args.output)
    except:
        print("Couldn't create the %s directory" % args.output)

for d in args.datasetdir:
    dname = path.basename(path.dirname(path.join(d,'docs')))
    doc_paths = sorted(glob(path.join(d,'docs','*')))
    doc_names = list(map(path.basename, doc_paths))
    y = load_labels(d)
    
    if not all(y[:,0]==doc_names):
        print("Dataset %s cant find all ids to doc.labeled" % d)
        break

    y = LabelEncoder().fit_transform(y[:,1])
    skf = StratifiedKFold(n_splits=args.nfolds)

    for w in args.window:
        docs = np.array(Document.load_path(doc_paths, w=w, verbose=not args.silence))
        for df in args.min_df:
            for m in args.metric:
                for f, (train_index, test_index) in enumerate(skf.split(docs, y)):
                    botg = BoTG(metric=m, min_df=df)
                    print("fold%d_%s_w%d_df%d_m-%s" % (f, dname,w,df,m))
                    docs_train, y_train = docs[train_index], y[train_index]
                    botg.fit(docs_train, verbose=not args.silence)
                    for p in args.pooling:
                        for a in args.assignment:
                            filename_train = "train%d_%s_w%d_df%d_m-%s_p-%s_a-%s" % (f,dname,w,df,m,p,a)
                            print("  %s" % filename_train)
                            X = botg.transform(docs_train, pooling=p, assignment=a, verbose=not args.silence)
                            dump_svmlight_file(X,y_train, path.join(args.output,filename_train))

                            docs_test, y_test = docs[test_index], y[test_index]
                            filename_test = "test%d_%s_w%d_df%d_m-%s_p-%s_a-%s" % (f,dname,w,df,m,p,a)
                            print("  %s" % filename_test)
                            X = botg.transform(docs_test, pooling=p, assignment=a, verbose=not args.silence)
                            dump_svmlight_file(X,y_test, path.join(args.output,filename_test))
