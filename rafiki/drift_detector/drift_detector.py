import time
import logging
import os
import traceback

import numpy as np
from rafiki.cache import Cache
from rafiki.db import Database
from rafiki.container import DockerSwarmContainerManager 

from .services_manager import ServicesManager

logger = logging.getLogger(__name__)

class Drift_Detector(object):
    def __init__(self, db=Database()):
        self._db = db

    def get_retrain_data_url(self):
        pass

    def create_drift_detection_service(self):
        service = self._services_manager.create_drift_detection_service()

        

        return {
            'id': service.id
        }

    def stop_drift_detection_service(self):
        service_id = self._services_manager.stop_drift_detection_service()

        return {
            'id': service_id
        }

    # # this is a sequential way to do detection for a trial_id
    # # TODO: make it multi thread
    # # naively users can send multiple posts to detect_with_detector_name()
    # def detect(self, trial_id):
    #     detector_subs = self._db.get_detector_subscriptions_by_trial_id(trail_id)
    #     for sub in detector_subs:
    #         self.detect_with_detector_name(trial_id, sub.detector_name)


    # def detect_with_detector_name(self, trial_id, detector_name):
    #     if detector_name not in self._drift_detectors:
    #         detector = self._db.get_detector_by_name(detector_name)
    #         clazz = load_detector_class(detector.detector_file_bytes, detector.detector_class)
    #         self._drift_detectors[detector_name] = clazz
        

    def subscribe_detector(self, trial_id, detector_name):
        detector_sub = self._db.create_detector_sub(
            trial_id=trial_id,
            detector_name=detector_name
        )

        self._db.commit()

        trial = self._db.get_trial(trial_id)
        trial = self._db.mark_trial_subscription_to_drift_detection_service(trial)
        self._db.commit()
        train_job = self._db.get_train_job(trial.train_job_id)
        train_job = self._db.mark_train_job_subscription_to_drift_detection_service(train_job)
        self._db.commit()
        
        return {
            'trial_id': trial_id,
            'name': detector_name
        }

    def create_detector(self, user_id, name, detector_file_bytes, detector_class):
        detector = self._db.create_detector(
            user_id=user_id,
            name=name,
            detector_file_bytes=detector_file_bytes,
            detector_class=detector_class
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
        