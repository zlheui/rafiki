import time
import logging
import os
import uuid
import traceback
import pprint
import numpy as np
from rafiki.db import Database

logger = logging.getLogger(__name__)


class Drift_Detector(object):
    def __init__(self, db=DataBase()):
        self._drift_detector = {}
        self._db = db

    def get_retrain_data_url(self):
        pass


    def create_detector(self, user_id, name, detector_file_bytes):
        detector = self._db.create_detector(
            user_id=user_id,
            name=name,
            detector_file_bytes=model_file_bytes
        )

        return {
            'name': detector.name 
        }

    def __enter__(self):
        self.connect()

    def connect(self):
        self._db.connect()

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()

    def disconnect(self):
        self._db.disconnect()
        