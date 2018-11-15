import tensorflow as tf
from tensorflow import keras
import json
import os
import tempfile
import numpy as np
import base64

from rafiki.model import BaseModel, InvalidModelParamsException, validate_model_class, load_dataset
from rafiki.constants import TaskType

class TfSingleHiddenLayerVector(BaseModel):
    '''
    Implements a fully-connected neural network with a single hidden layer on Tensorflow
    '''

    def get_knob_config(self):
        return {
            'knobs': {
                'hidden_layer_units': {
                    'type': 'int',
                    'range': [2, 128]
                },
                'epochs': {
                    'type': 'int',
                    'range': [1, 20]
                },
                'learning_rate': {
                    'type': 'float_exp',
                    'range': [1e-5, 1e-1]
                },
                'batch_size': {
                    'type': 'int_cat',
                    'values': [1, 2, 4, 8, 16, 32, 64, 128]
                }
            }
        }

    def get_predict_label_mapping(self):
        return self._predict_label_mapping
        
    def init(self, knobs):
        self._batch_size = knobs.get('batch_size')
        self._epochs = knobs.get('epochs')
        self._hidden_layer_units = knobs.get('hidden_layer_units')
        self._learning_rate = knobs.get('learning_rate')

        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        
    def train(self, dataset_uri, task):
        (images, labels) = self._load_dataset(dataset_uri, task)

        class_names = np.unique(labels)
        num_classes = len(class_names)
        self._predict_label_mapping = dict(zip(range(num_classes), class_names))
        train_and_evalutate_label_mapping = {v: k for k, v in  self._predict_label_mapping.items()}

        labels = np.array([train_and_evalutate_label_mapping[label] for label in labels])

        with self._graph.as_default():
            self._model = self._build_model(num_classes)
            with self._sess.as_default():
                self._model.fit(
                    images, 
                    labels, 
                    epochs=self._epochs, 
                    batch_size=self._batch_size
                )

    def evaluate(self, dataset_uri, task):
        (images, labels) = self._load_dataset(dataset_uri, task)
        train_and_evalutate_label_mapping = {v: k for k, v in  self._predict_label_mapping.items()}
        labels = np.array([train_and_evalutate_label_mapping[label] for label in labels])

        with self._graph.as_default():
            with self._sess.as_default():
                (loss, accuracy) = self._model.evaluate(images, labels)
        return accuracy

    def predict(self, queries):
        X = np.array(queries)
        with self._graph.as_default():
            with self._sess.as_default():
                probs = self._model.predict(X)
        return probs

    def destroy(self):
        self._sess.close()

    def dump_parameters(self):
        # TODO: Not save to & read from a file 

        # Save whole model to temp h5 file
        tmp = tempfile.NamedTemporaryFile(delete=False)
        with self._graph.as_default():
            with self._sess.as_default():
                self._model.save(tmp.name)
        
        # Read from temp h5 file & encode it to base64 string
        with open(tmp.name, 'rb') as f:
            h5_model_bytes = f.read()

        h5_model_base64 = base64.b64encode(h5_model_bytes).decode('utf-8')

        # Remove temp file
        os.remove(tmp.name)

        return {
            'h5_model_base64': h5_model_base64,
            'predict_label_mapping': self._predict_label_mapping
        }

    def load_parameters(self, params):
        h5_model_base64 = params.get('h5_model_base64', None)

        if h5_model_base64 is None:
            raise InvalidModelParamsException()

        # TODO: Not save to & read from a file 

        # Convert back to bytes & write to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False)
        h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
        with open(tmp.name, 'wb') as f:
            f.write(h5_model_bytes)

        # Load model from temp file
        with self._graph.as_default():
            with self._sess.as_default():
                self._model = keras.models.load_model(tmp.name)
                
        # Remove temp file
        os.remove(tmp.name)

        if 'predict_label_mapping' in params:
            self._predict_label_mapping = params['predict_label_mapping']

    def _load_dataset(self, dataset_uri, task):
        # Here, we use Rafiki's in-built dataset loader
        return load_dataset(dataset_uri, task) 

    def _build_model(self, num_classes):
        hidden_layer_units = self._hidden_layer_units
        learning_rate = self._learning_rate

        model = keras.Sequential()
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(
            hidden_layer_units,
            activation=tf.nn.relu
        ))
        model.add(keras.layers.Dense(
            num_classes, 
            activation=tf.nn.softmax
        ))
        
        model.compile(
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


if __name__ == '__main__':
    validate_model_class(
        model_class=TfSingleHiddenLayerVector,
        train_dataset_uri='https://github.com/sdragon007/sea_dataset/blob/master/sea_train.zip?raw=true',
        test_dataset_uri='https://github.com/sdragon007/sea_dataset/blob/master/sea_test.zip?raw=true',
        task=TaskType.FEATURE_VECTOR_CLASSIFICATION,
        queries=[
            [3,4,5]
        ]
    )
