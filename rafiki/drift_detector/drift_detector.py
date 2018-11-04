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
    def __init__(self, db=Database()):
        self._drift_detector = {}
        self._db = db

    def get_retrain_data_url(self):
        pass

    def detect(self, trial_id):
        pass

    def detect_with_detector_name(self, trial_id, detector_name):
        pass


    def subscribe_detector(self, trial_id, detector_name):
        detector_sub = self._db.create_detector_sub(
            trial_id=trial_id,
            detector_name=detector_name
        )

        self._db.commit()

        trial = self._db.get_trial(trial_id)
        train_job = self._db.get_train_job(trial.train_job_id)

        trial = self._db.mark_trial_subscription_to_drift_detection_service(trial)
        train_job = self._db.mark_train_job_subscription_to_drift_detection_service(train_job)

        self._db.commit()
        
        return {
            'trial_id': trial_id,
            'name': detector_name
        }

    def create_detector(self, user_id, name, detector_file_bytes):
        detector = self._db.create_detector(
            user_id=user_id,
            name=name,
            detector_file_bytes=detector_file_bytes
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
        