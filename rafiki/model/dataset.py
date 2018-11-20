import numpy as np
from PIL import Image
import requests

from urllib.parse import urlparse
import zipfile
import io

from rafiki.constants import TaskType, DatasetProtocol

def load_dataset(uri, task):
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
                            encoded = io.BytesIO(dataset.read(entry))
                            image = np.array(Image.open(encoded))
                            if len(image.shape) != 2:
                                continue
                            label = entry.split('/')[-2]
                            labels.append(label)
                            images.append(image)
                return (np.array(images), np.array(labels))
            else:
                raise Exception('{} compression not supported'.format(uri))
        elif _is_feature_vector_classification(task):
            if _is_zip(uri):
                r = requests.get(uri)
                features = []
                labels = []
                with zipfile.ZipFile(io.BytesIO(r.content)) as dataset:
                    for entry in dataset.namelist():
                        if entry.endswith('.csv') or entry.endswith('.data'):
                            encoded = io.BytesIO(dataset.read(entry))
                            if '\r\n' in encoded.getvalue().decode('utf-8'):
                                delimiter = '\r\n'
                            else:
                                delimiter = '\n'
                            data_decoded = encoded.getvalue().decode('utf-8').split(delimiter)
                            data_split = [row.split(",") for row in data_decoded if row != '']
                            labels.extend(np.array([row[-1] for row in data_split]))
                            features.extend(np.array([row[:-1] for row in data_split]).astype(np.float))
                return (np.array(features), np.array(labels))
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
