import os
import numpy as np
from importlib import import_module
import uuid

TEMP_MODEL_FILE_NAME = 'tempModel'
TEMP_DETECTOR_FILE_NAME = 'tempDetector'

def generate_uuid():
    return str(uuid.uuid4())

def load_model_class(model_file_bytes, model_class):
    # Save the model file to disk
    tempFname = '{}_{}'.format(TEMP_MODEL_FILE_NAME, generate_uuid()) 
    f = open('{}.py'.format(tempFname), 'wb')
    f.write(model_file_bytes)
    f.close()

    # Import model file as module
    mod = import_module(tempFname)
    # Extract model class from module
    clazz = getattr(mod, model_class)

    # Remove temporary file
    os.remove(f.name)

    return clazz

def load_detector_class(detector_file_bytes, detector_class):
    # Save the detector file to disk
    tempFname = '{}_{}'.format(TEMP_DETECTOR_FILE_NAME, generate_uuid()) 
    f = open('{}.py'.format(tempFname), 'wb')
    f.write(detector_file_bytes)
    f.close()

    # Import detector file as module
    det = import_module(tempFname)

    # Extract detector class from module
    clazz = getattr(det, detector_class)

    # Remove temporary file
    os.remove(f.name)

    return clazz

def parse_model_prediction(prediction):
    if isinstance(prediction, np.ndarray):
        return prediction.tolist()

    return prediction