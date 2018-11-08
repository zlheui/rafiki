import time
import logging
import os
import traceback
import pprint
from multiprocessing import Process

from rafiki.utils.model import load_detector_class
from rafiki.db import Database
from rafiki.cache import Cache
from rafiki.config import DRIFT_WORKER_SLEEP, DRIFT_DETECTION_BATCH_SIZE
from rafiki.constants import ServiceType


def update_on_feedbacks(clazz, train_job_id, feedbacks):
    detector_inst = clazz()
    detector_inst.init()

    while True:
        try:
            detector_inst.update_on_feedbacks(train_job_id, feedbacks)
        except:
            time.sleep(DRIFT_WORKER_SLEEP)
            continue
        else:
            break


class DriftDetectionFeedbackWorker(Object):
    def __init__(self, service_id, cache=Cache(), db=Database(isolation_level='REPEATABLE_READ')):
        self._cache = cache
        self._db = db
        self._service_id = service_id
        self._detectors = {}

    def start(self):
        logger.info('Starting drift detection worker for service of id {}...' \
            .format(self._service_id))

        # Add to set of running workers
        self._cache.add_drift_detection_worker(self._service_id, ServiceType.DRIFT_FEEDBACK)

        while True:
            (feedback_ids, train_job_ids, query_indexes, labels) = \
                self._cache.pop_feedbacks_of_worker(self._service_id, DRIFT_DETECTION_BATCH_SIZE)

            if len(labels) > 0:
                logger.info('Detecting concept drift for feedbacks...')
                logger.info(['{}_{}_{}'.format(a,b,c) for a,b,c in zip(train_job_ids, query_indexes, labels)])

                train_job_id_to_feedbacks = {}
                for (train_job_id, query_index, label) in zip(train_job_ids, query_indexes, labels):
                    if train_job_id in train_job_id_to_feedbacks:
                        train_job_id_to_feedbacks[train_job_id].append((query_index, label))
                    else:
                        train_job_id_to_feedbacks[train_job_id] = [(query_index, label)]

                train_job_id_to_detection_methods = {}
                detection_methods = []
                self._db.connect()
                for train_job_id,_ in train_job_id_to_feedbacks.items():
                    trials = self._db.get_trials_of_train_job()
                    train_job_id_to_detection_methods[train_job_id] = []
                    for trial in trials:
                        detector_subs = self._db.get_detector_subscriptions_by_trial_id(trial.id)
                        for sub in detector_subs:
                            if sub.name not in train_job_id_to_detection_methods[train_job_id]:
                                train_job_id_to_detection_methods[train_job_id].append(sub.name)
                            if sub.name not in self._detectors and sub.name not in detection_methods:
                                detection_methods.append(sub.name)
                self._db.commit()

                # load new detection classes
                for detector_name in detection_methods:
                    detector = self._db.get_detector_by_name(detector_name)
                    clazz = load_detector_class(detector.detector_file_bytes, detector.detector_class)
                    self._drift_detectors[detector_name] = clazz
                self._db.commit()

                procs = []
                for train_job_id,feedbacks in train_job_id_to_feedbacks.items():
                    for detector_method in train_job_id_to_detection_methods[train_job_id]:
                        proc = Process(target=update_on_feedbacks, args=(self._detectors[detector_method], train_job_id, feedbacks))
                        procs.append(proc)
                        proc.start()
                for proc in procs:
                    proc.join()
        
        time.sleep(DRIFT_WORKER_SLEEP)

    def stop(self):
        # Remove from set of running workers
        self._cache.delete_drift_detection_worker(self._service_id, ServiceType.DRIFT_FEEDBACK)