from utils import Dataset
from TFDFEmb import AttentionTFIDFClassifier



d = Dataset('/home/mangaravite/Documentos/datasets/classification/datasets/acm')
fold = next(d.get_fold_instances(10, with_val=True))

class_att = AttentionTFIDFClassifier(batch_size=32, _verbose=True)
class_att.fit( fold.X_train, fold.y_train, fold.X_val, fold.y_val )
