import numpy as np
from PIL import Image
import requests

from urllib.parse import urlparse
import zipfile
import io

from rafiki.constants import TaskType, DatasetProtocol
from rafiki.utils.model import parse_model_prediction

def load_dataset_training(uri, task, model = None):
    parsed_uri = urlparse(uri)
    protocol = '{uri.scheme}'.format(uri=parsed_uri)
    if _is_http(protocol) or _is_https(protocol):
        if _is_image_classification(task):
            if _is_zip(uri):
                r = requests.get(uri)
                images = []
                labels = []
                with zipfile.ZipFile(io.BytesIO(r.content)) as dataset:
                    for entry in dataset.namelist():
                        if entry.endswith('.png') or entry.endswith('.jpg') or entry.endswith('.jpeg'):
                            label = entry.split('/')[-2]
                            labels.append(label)
                            encoded = io.BytesIO(dataset.read(entry))
                            image = np.array(Image.open(encoded))
                            images.append(image)
                images = np.array(images)
                labels = np.array(labels)
                predictions = None
                if len(images) > 0 and model is not None:
                    try:
                        predictions = self._model.predict(queries)
                        predictions = [parse_model_prediction(x) for x in predictions]
                        predictions = np.array(np.argmax(predictions, axis=0))
                    except Exception:
                        pass

                return (images, labels, predictions)
            else:
                raise Exception('{} compression not supported'.format(uri))
        elif _is_feature_vector_classification(task):
            if _is_zip(uri):
                r = requests.get(uri)
                features = []
                labels = []
                with zipfile.ZipFile(io.BytesIO(r.content)) as dataset:
                    for entry in dataset.namelist():
                        encoded = io.BytesIO(dataset.read(entry))
                        if '\r\n' in encoded.getvalue().decode('utf-8'):
                            delimiter = '\r\n'
                        else:
                            delimiter = '\n'
                        data_decoded = encoded.getvalue().decode('utf-8').split(delimiter)
                        data_split = [row.split(",") for row in data_decoded if row != '']
                        labels.extend(np.array([row[-1] for row in data_split]))
                        features.extend(np.array([row[:-1] for row in data_split]).astype(np.float))
                features = np.array(features)
                labels = np.array(labels)
                predictions = None
                if len(features) > 0 and model is not None:
                    try:
                        predictions = self._model.predict(features)
                        predictions = [parse_model_prediction(x) for x in predictions]
                        predictions = np.array(np.argmax(predictions, axis=0))
                    except Exception:
                        pass

                return (features, labels, predictions)
            else:
                raise Exception('{} compression not supported'.format(uri))
        else:
            raise Exception('{} task not supported'.format(task))
    else:
        raise Exception('Dataset URI scheme not supported: {}'.format(protocol))

def _is_zip(uri):
    return '.zip' in uri

def _is_http(protocol):
    return protocol == DatasetProtocol.HTTP

def _is_https(protocol):
    return protocol == DatasetProtocol.HTTPS

def _is_image_classification(task):
    return task == TaskType.IMAGE_CLASSIFICATION

def _is_feature_vector_classification(task):
    return task == TaskType.FEATURE_VECTOR_CLASSIFICATION
