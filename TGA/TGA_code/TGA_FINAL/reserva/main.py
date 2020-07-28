from utils import Dataset
from TGA_model import TGA
from tqdm import tqdm
import numpy as np
import warnings
from itertools import product
warnings.filterwarnings('ignore')
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument('-c', '--cuda', default=0, type=int)

args = parser.parse_args()

dataset = Dataset(args.dataset)

w=5
nh=4
lr=1e-3
l2=1e-3
bs=16

config_name = f"w={w}_nh={nh}_lr={lr:.5f}_l2={l2:.5f}_bs={bs}"
print(config_name, flush=True)
filename = dataset.dname + '_' + config_name + '.json'

print(f"############## {dataset.dname} ##############", flush=True)
tga = TGA(n_epochs=50, lr=lr, train_batch_size=bs, w=w, n_heads=nh, weight_decay=l2, device=f"cuda:{args.cuda}",
            pretrained_vec='/home/mangaravite/Documents/pretrained_vectors/glove.6B.300d.txt',
            verbose=True)
params = tga.get_params()
folds = []
for f, fold in enumerate(dataset.get_fold_instances(10)):
    print(f"Fold {f}", flush=True)
    tga.fit( fold.X_train, fold.y_train, fold.X_val, fold.y_val )
    y_pred = tga.predict( fold.X_test )
    correct = sum([ p==l for (p,l) in zip(y_pred,fold.y_test) ])
    total = len(fold.X_test)
    fold_out = { 'y_pred': list(map(int, list(y_pred))),
                 'y_test': list(map(int, list(fold.y_test))),
                 'epochs_acc': list(tga.checkpoints),
                 'model': tga.path_to_save
                }
    folds.append( fold_out )
    print( f"END FOLD: ACC={(correct/total):.4f}" )
    params['folds'] = folds
    with open(filename, 'w') as outfile:
        json.dump(params, outfile)

