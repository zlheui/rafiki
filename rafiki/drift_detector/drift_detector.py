import time
import logging
import os
import traceback

from rafiki.db import Database
from rafiki.container import DockerSwarmContainerManager 
from .services_manager import ServicesManager

logger = logging.getLogger(__name__)

class Drift_Detector(object):
    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._services_manager = ServicesManager(db, container_manager)

    def get_retrain_data_url(self):
        pass

    def create_drift_detection_service(self, service_type):
        service = self._services_manager.create_drift_detection_service(service_type)

        return {
            'id': service.id
        }

    def stop_drift_detection_service(self, service_type):
        service_id = self._services_manager.stop_drift_detection_service(service_type)

        return {
            'id': service_id
        }

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
        