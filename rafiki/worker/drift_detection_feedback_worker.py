import time
import logging
import os
import traceback
import pprint
from multiprocessing import Process

from rafiki.utils.drift_detection_method import load_detector_class
from rafiki.db import Database
from rafiki.cache import Cache
from rafiki.config import DRIFT_WORKER_SLEEP, DRIFT_DETECTION_BATCH_SIZE, WAIT_FOR_FEEDBACK_SLEEP, MIN_FEEDBACK_COUNT
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.constants import ServiceType
from rafiki.client import Client

logger = logging.getLogger(__name__)

class DriftDetectionFeedbackWorker(object):
    def __init__(self, service_id, cache=Cache(), db=Database(isolation_level='REPEATABLE_READ')):
        self._cache = cache
        self._db = db
        self._db.connect()
        self._service_id = service_id
        self._detectors = {}
        self._client = None

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
                #feedback becomes a tuple of (query_index, label)
                for (train_job_id, query_index, label) in zip(train_job_ids, query_indexes, labels):
                    if train_job_id in train_job_id_to_feedbacks:
                        train_job_id_to_feedbacks[train_job_id].append((query_index, label))
                    else:
                        train_job_id_to_feedbacks[train_job_id] = [(query_index, label)]

                train_job_id_to_detection_methods = {}
                train_job_id_to_trial_ids = {}
                detection_methods = []
                
                for train_job_id,_ in train_job_id_to_feedbacks.items():
                    trials = self._db.get_trials_of_train_job(train_job_id)
                    train_job_id_to_trial_ids[train_job_id] = [trial.id for trial in trials]
                    train_job_id_to_detection_methods[train_job_id] = []
                    for trial in trials:
                        detector_subs = self._db.get_detector_subscriptions_by_trial_id(trial.id)
                        for sub in detector_subs:
                            if sub.detector_name not in train_job_id_to_detection_methods[train_job_id]:
                                train_job_id_to_detection_methods[train_job_id].append(sub.detector_name)
                            if sub.detector_name not in self._detectors and sub.detector_name not in detection_methods:
                                detection_methods.append(sub.detector_name)
                self._db.commit()

                logger.info('load new detection classes')
                # load new detection classes
                for detector_name in detection_methods:
                    detector = self._db.get_detector_by_name(detector_name)
                    clazz = load_detector_class(detector.detector_file_bytes, detector.detector_class)
                    self._detectors[detector_name] = clazz
                self._db.commit()

                logger.info('start multiprocessing')
                procs = []
                for train_job_id,feedbacks in train_job_id_to_feedbacks.items():
                    for detector_method in train_job_id_to_detection_methods[train_job_id]:
                        proc = Process(target=self._update_on_feedbacks, args=(detector_method, self._detectors[detector_method], \
                                  train_job_id, train_job_id_to_trial_ids[train_job_id], feedbacks, logger))
                        procs.append(proc)
                        proc.start()
                for proc in procs:
                    proc.join()

                logger.info('finish multiprocessing')
                
            time.sleep(DRIFT_WORKER_SLEEP)

    def stop(self):
        # Remove from set of running workers
        self._cache.delete_drift_detection_worker(self._service_id, ServiceType.DRIFT_FEEDBACK)
        self._db.disconnect()

    def _update_on_feedbacks(self, detector_name, clazz, train_job_id, trial_ids, feedbacks, logger):
        logger.info('detect real concept drift')
        logger.info('feedbacks: {}'.format(feedbacks))
        #try 5 times only and exit if still fail
        for i in range(5):
            try:
                for trial_id in trial_ids:
                    logger.info(trial_id)
                    if not self._db.trial_subscribed_to_detector(trial_id, detector_name):
                        logger.info('{} is not subscribed to {} detector, skipping drift detection'.format(trial_id, detector_name))
                        continue
                    else:
                        logger.info('trial: {}, detector: {}'.format(trial_id, detector_name))
                    detector_inst = clazz()
                    detector_inst.init(ServiceType.DRIFT_FEEDBACK, detector_name, trial_id, logger=logger)
                    detection_result, index_of_change = detector_inst.update_on_feedbacks(trial_id, feedbacks, logger)
                    param_str = detector_inst.dump_parameters(logger)
                    self._db.update_trial_detector_param(trial_id, detector_name, param_str)

                    if detection_result and index_of_change is not None and detection_result == True:
                        logger.info('Drift is detected at query index {} with {} trend for {}th class'.format(index_of_change, \
                                             detector_inst._param['trend'], detector_inst._param['index']))
                        trial = self._db.get_trial(trial_id)
                        train_job_id = trial.train_job_id
                        train_job = self._db.get_train_job(train_job_id)

                        if train_job.retrain_scheduled == 'True':
                            logger.info('retraining already scheduled for train job id: {}'.format(train_job_id))
                            break
                        self._db.mark_train_job_retrain_scheduled(train_job)

                        #wait until have enough feedback to start a retraining
                        try:
                            while True:
                                self._db.commit()
                                feedback_count = self._db.get_number_feedback_after_index_by_train_job(train_job_id, index_of_change)
                                if  feedback_count < MIN_FEEDBACK_COUNT:
                                    time.sleep(WAIT_FOR_FEEDBACK_SLEEP)
                                else:
                                    if self._client is None:
                                        self._client = self._make_client()
                                    res = self._client.create_retrain_service(train_job_id, index_of_change)
                                    logger.info('retrain status: {}'.format(res))
                                    if bool(res['created']):
                                        break
                                    else:
                                        self._db.unmark_train_job_retrain_scheduled(train_job)
                        except Exception as e:
                            logger.error(e, exc_info=True)
                            self._db.unmark_train_job_retrain_scheduled(train_job)
            except Exception as e:
                logger.error(e, exc_info=True)
                time.sleep(DRIFT_WORKER_SLEEP)
                continue
            else:
                break

    def _make_client(self):
        admin_host = os.environ['ADMIN_HOST']
        admin_port = os.environ['ADMIN_PORT']
        advisor_host = os.environ['ADVISOR_HOST']
        advisor_port = os.environ['ADVISOR_PORT']
        data_repository_host = os.environ['DATA_REPOSITORY_HOST']
        data_repository_port = os.environ['DATA_REPOSITORY_PORT']
        superadmin_email = SUPERADMIN_EMAIL
        superadmin_password = SUPERADMIN_PASSWORD
        client = Client(admin_host=admin_host, 
                        admin_port=admin_port, 
                        advisor_host=advisor_host,
                        advisor_port=advisor_port,
                        data_repository_host=data_repository_host,
                        data_repository_port=data_repository_port)
        client.login(email=superadmin_email, password=superadmin_password)
        return client