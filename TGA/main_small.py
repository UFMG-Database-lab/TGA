from utils import Dataset
from TFDFEmb import AttentionTFIDFClassifier
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials, STATUS_FAIL #, SparkTrials, STATUS_FAIL

from sklearn.metrics import f1_score

from os import path, mkdir
from multiprocessing import cpu_count

import sys

d = Dataset(sys.argv[1])

path_result = path.join('result', d.dname)
if not path.exists(path_result):
    mkdir(path_result)

for (i,fold) in enumerate(d.get_fold_instances(10, with_val=True)):
    class_att = AttentionTFIDFClassifier(nepochs=150, _verbose=True)
    print(class_att)
    class_att.fit( fold.X_train, fold.y_train, fold.X_val, fold.y_val )
    y_pred = class_att.predict( fold.X_test )

    with open(path.join(path_result, f'fold{i}'), 'w') as file_writer:
        file_writer.write( ';'.join( map(str, y_pred) ) )

    acc = ( y_pred == fold.y_test ).sum() / len(y_pred)
    f1_mic = f1_score(fold.y_test, y_pred, average='micro')
    f1_mac = f1_score(fold.y_test, y_pred, average='macro')

    print(f"fold {i} ACC={acc} f1_mi={f1_mic} f1_ma={f1_mac}")



    


