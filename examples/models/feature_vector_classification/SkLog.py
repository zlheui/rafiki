from sklearn.linear_model import LogisticRegression
import json
import pickle
import os
import base64
import numpy as np

from rafiki.model import BaseModel, InvalidModelParamsException, validate_model_class, load_dataset
from rafiki.constants import TaskType

class SkLogVector(BaseModel):
    '''
    Implements a logistic regression on scikit-learn
    '''

    def get_knob_config(self):
        return {
            'knobs': {
                'max_iter': {
                    'type': 'int',
                    'range': [10, 200]
                },
                'penalty': {
                    'type': 'string',
                    'values': ['l1', 'l2']
                },
                'tol': {
                    'type': 'float_exp',
                    'range': [1e-5, 1e-1]
                },
                'C': {
                    'type': 'float_exp',
                    'range': [1e-2, 1e2]
                }
            }
        }

    def get_predict_label_mapping(self):
        return self._predict_label_mapping

    def init(self, knobs):
        self._max_iter = knobs.get('max_iter') 
        self._penalty = knobs.get('penalty') 
        self._tol = knobs.get('tol') 
        self._C = knobs.get('C') 
        self._clf = self._build_classifier(
            self._max_iter,
            self._penalty,
            self._tol,
            self._C
        )
        
    def train(self, dataset_uri, task):
        (images, labels) = self._load_dataset(dataset_uri, task)
        class_names = np.unique(labels)
        num_classes = len(class_names)
        self._predict_label_mapping = dict(zip(range(num_classes), class_names))
        train_and_evalutate_label_mapping = {v: k for k, v in  self._predict_label_mapping.items()}

        labels = np.array([train_and_evalutate_label_mapping[label] for label in labels])

        X = self._prepare_X(images)
        y = labels
        self._clf.fit(X, y)

    def evaluate(self, dataset_uri, task):
        (images, labels) = self._load_dataset(dataset_uri, task)
        train_and_evalutate_label_mapping = {v: k for k, v in  self._predict_label_mapping.items()}
        labels = np.array([train_and_evalutate_label_mapping[label] for label in labels])
        
        X = self._prepare_X(images)
        y = labels
        preds = self._clf.predict(X)
        accuracy = sum(y == preds) / len(y)
        return accuracy

    def predict(self, queries):
        X = self._prepare_X(queries)
        probs = self._clf.predict_proba(X)
        return probs

    def destroy(self):
        pass

    def dump_parameters(self):
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        return {
            'clf_base64': clf_base64,
            'predict_label_mapping': self._predict_label_mapping
        }

    def load_parameters(self, params):
        if 'clf_base64' in params:
            clf_bytes = base64.b64decode(params['clf_base64'].encode('utf-8'))
            self._clf = pickle.loads(clf_bytes)

        if 'predict_label_mapping' in params:
            self._predict_label_mapping = params['predict_label_mapping']

    def _prepare_X(self, images):
        return [np.array(image).flatten() for image in images]

    def _load_dataset(self, dataset_uri, task):
        # Here, we use Rafiki's in-built dataset loader
        return load_dataset(dataset_uri, task) 

    def _build_classifier(self, max_iter, penalty, tol, C):
        clf = LogisticRegression(
            max_iter=max_iter,
            penalty=penalty,
            tol=tol,
            C=C
        ) 
        return clf


if __name__ == '__main__':
    validate_model_class(
        model_class=SkLogVector,
        train_dataset_uri='https://github.com/sdragon007/sea_dataset/blob/master/sea_train.zip?raw=true',
        test_dataset_uri='https://github.com/sdragon007/sea_dataset/blob/master/sea_test.zip?raw=true',
        task=TaskType.FEATURE_VECTOR_CLASSIFICATION,
        queries=[
            [5.334, 6.2113, 3.451]
        ]
    )
