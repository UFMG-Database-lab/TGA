from utils import Dataset
from TFDFEmb import AttentionTFIDFClassifier
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials, STATUS_FAIL

import sys

d = Dataset(sys.argv[1])

space = {
    "batch_size": hp.quniform("batch_size", 8, 128, 32),
    "k": hp.quniform("k", 64, 512, 64),
    "max_drop": hp.uniform("max_drop", 0.3, 0.9),
    "mindf": hp.quniform("mindf", 1, 4, 1),
    "stopwords": hp.choice("stopwords", ["nltk", "sklearn", None]),
}

for (i,fold) in enumerate(d.get_fold_instances(10, with_val=True)):
    def hyperparameter_tuning_try(params):
        try:
            class_att = AttentionTFIDFClassifier(**params, nepochs=25, _verbose=False)
            print(class_att)
            class_att.fit( fold.X_train, fold.y_train, fold.X_val, fold.y_val )
            return {"loss": class_att._loss, "status": STATUS_OK }
        except:
            return { "status": STATUS_FAIL }
    
    trials = Trials()
    
    best = fmin(
        fn=hyperparameter_tuning_try,
        space = space, 
        algo=tpe.suggest, 
        max_evals=15, 
        trials=trials
    )

    print("Best: {}".format(best))

    
    params = [ (label, value) if label != 'stopwords' else (label,["nltk", "sklearn", None][value]) for (label,value) in best.items() ]
    params = dict( params )
    class_att = AttentionTFIDFClassifier(**params, _verbose=True)
    class_att.fit( fold.X_train, fold.y_train, fold.X_val, fold.y_val )

    y_pred = class_att.predict( fold.X_test )

    print(f"ACC_test fold {i}: {( y_pred == fold.y_test ).sum() / len(y_pred)}")



    


