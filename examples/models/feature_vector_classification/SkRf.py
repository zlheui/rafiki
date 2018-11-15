from sklearn.ensemble import RandomForestClassifier
import json
import pickle
import os
import base64
import numpy as np

from rafiki.model import BaseModel, InvalidModelParamsException, validate_model_class, load_dataset
from rafiki.constants import TaskType

class SkRfVector(BaseModel):
    '''
    Implements a random forest classifier on scikit-learn
    '''

    def get_knob_config(self):
        return {
            'knobs': {
                'n_estimators': {
                    'type': 'int',
                    'range': [1, 20]
                },
                'max_depth': {
                    'type': 'int',
                    'range': [2, 20]
                },
                'criterion': {
                    'type': 'string',
                    'values': ['gini', 'entropy']
                },
                'min_samples_split': {
                    'type': 'int',
                    'range': [2, 5]
                },
                'bootstrap': {
                    'type': 'bool',
                    'values': [True, False]
                }
            }
        }

    def get_predict_label_mapping(self):
        return self._predict_label_mapping

    def init(self, knobs):
        self._n_estimators = knobs.get('n_estimators') 
        self._max_depth = knobs.get('max_depth') 
        self._criterion = knobs.get('criterion') 
        self._min_samples_split = knobs.get('min_samples_split') 
        self._bootstrap = knobs.get('bootstrap') 
        self._clf = self._build_classifier(
            self._n_estimators,
            self._max_depth,
            self._criterion,
            self._min_samples_split,
            self._bootstrap
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

    def _build_classifier(self, n_estimators, max_depth, criterion, min_samples_split, bootstrap):
        clf = RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            criterion = criterion,
            min_samples_split = min_samples_split,
            bootstrap = bootstrap
        )
        return clf

if __name__ == '__main__':
    validate_model_class(
        model_class=SkRfVector,
        train_dataset_uri='https://github.com/sdragon007/sea_dataset/blob/master/sea_train.zip?raw=true',
        test_dataset_uri='https://github.com/sdragon007/sea_dataset/blob/master/sea_test.zip?raw=true',
        task=TaskType.FEATURE_VECTOR_CLASSIFICATION,
        queries=[
            [5.334, 6.2113, 3.451]
        ]
    )
