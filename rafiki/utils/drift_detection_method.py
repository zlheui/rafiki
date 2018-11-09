import os
import numpy as np
from importlib import import_module

TEMP_DETECTOR_FILE_NAME = 'temp'

def load_detector_class(detector_file_bytes, detector_class):
    # Save the detector file to disk
    f = open('{}.py'.format(TEMP_DETECTOR_FILE_NAME), 'wb')
    f.write(detector_file_bytes)
    f.close()

    # Import detector file as module
    det = import_module(TEMP_DETECTOR_FILE_NAME)

    # Extract detector class from module
    clazz = getattr(det, detector_class)

    # Remove temporary file
    os.remove(f.name)

    return clazz