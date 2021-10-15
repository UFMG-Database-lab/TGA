from utils import Dataset
from TFDFEmb import AttentionTFIDFClassifier
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials, STATUS_FAIL

d = Dataset('/home/mangaravite/Documentos/datasets/classification/datasets/acm')

space = {
    "batch_size": hp.quniform("batch_size", 8, 128, 8),
    "k": hp.quniform("k", 32, 512, 32),
    "max_drop": hp.uniform("max_drop", 0.3, 0.9),
    "mindf": hp.quniform("mindf", 1, 5, 1),
    "stopwords": hp.choice("stopwords", ["nltk", "sklearn", None]),
}

for fold in d.get_fold_instances(10, with_val=True):
    def hyperparameter_tuning_try(params):
        try:
            class_att = AttentionTFIDFClassifier(**params, _verbose=True)
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
        max_evals=1, 
        trials=trials
    )

    print("Best: {}".format(best))

