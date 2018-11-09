import time
import logging
import os
import traceback
import pprint
from multiprocessing import Process
from multiprocessing import Queue

from rafiki.utils.drift_detection_method import load_detector_class
from rafiki.db import Database
from rafiki.cache import Cache
from rafiki.config import DRIFT_WORKER_SLEEP, DRIFT_DETECTION_BATCH_SIZE
from rafiki.constants import ServiceType

logger = logging.getLogger(__name__)

class DriftDetectionQueryWorker(object):
    def __init__(self, service_id, cache=Cache(), db=Database(isolation_level='REPEATABLE_READ')):
        self._cache = cache
        self._db = db
        self._service_id = service_id
        self._detectors = {}
        self._client = None

    def start(self):
        logger.info('Starting drift detection worker for service of id {}...' \
            .format(self._service_id))

        # Add to set of running workers
        self._cache.add_drift_detection_worker(self._service_id, ServiceType.DRIFT_QUERY)

        while True:
            (query_ids, train_job_ids, queries) = \
                self._cache.pop_queries_of_drift_detection_worker(self._service_id, DRIFT_DETECTION_BATCH_SIZE)

            if len(queries) > 0:
                logger.info('Detecting concept drift for queries...')
                logger.info(['{}_{}'.format(a,b) for a,b in zip(train_job_ids, queries)])

                train_job_id_to_queries = {}
                for (train_job_id, query_id, query) in zip(train_job_ids, query_ids, queries):
                    if train_job_id in train_job_id_to_queries:
                        train_job_id_to_queries[train_job_id].append((query_id, query))
                    else:
                        train_job_id_to_queries[train_job_id] = [(query_id, query)]

                train_job_id_to_detection_methods = {}
                detection_methods = []
                self._db.connect()
                for train_job_id,_ in train_job_id_to_queries.items():
                    trials = self._db.get_trials_of_train_job(train_job_id)
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

                logger.info('multiprocessing for uploading datasets')
                procs = []
                output = Queue()
                for train_job_id,job_queries in train_job_id_to_queries.items():
                    if len(train_job_id_to_detection_methods[train_job_id]) > 0:
                        tmp_detector_method = train_job_id_to_detection_methods[train_job_id][0]
                        proc = Process(target=self._upload_queries, args=(self._detectors[tmp_detector_method], train_job_id, job_queries, output, logger))
                        procs.append(proc)
                        proc.start()

                results = [output.get() for p in procs]
                for proc in procs:
                    proc.join()

                logger.info(results)

                train_job_id_to_query_index = {}
                for e in results:
                    train_job_id_to_query_index[e[0]] = int(e[1])

                logger.info('finish uploading datasets')

                logger.info('multiprocessing for drift detection')
                procs = []
                for train_job_id,queries in train_job_id_to_queries.items():
                    for detector_method in train_job_id_to_detection_methods[train_job_id]:
                        proc = Process(target=self._update_on_queries, \
                         args=(self._detectors[detector_method], train_job_id, queries, train_job_id_to_query_index[train_job_id], logger))
                        procs.append(proc)
                        proc.start()
                for proc in procs:
                    proc.join()
                logger.info('finish drift detection')

            time.sleep(DRIFT_WORKER_SLEEP)

    def stop(self):
        # Remove from set of running workers
        self._cache.delete_drift_detection_worker(self._service_id, ServiceType.QUERY)

    def _update_on_queries(self, clazz, train_job_id, queries, query_index, logger):
        detector_inst = clazz()
        detector_inst.init()

        logger.info('detect covariate drift')

        while True:
            try:
                detection_result, index_of_change = detector_inst.update_on_queries(train_job_id, queries, query_index)
                if detection_result and index_of_change is not None:
                    if self._client is None:
                        self._client = self._make_client()

                    res = self._client.create_new_dataset(train_job_id, index_of_change)
                    if bool(res['created']):
                        # TODO: schedule admin to retrain the trail
                        pass

            except:
                time.sleep(DRIFT_WORKER_SLEEP)
                continue
            else:
                break

    def _upload_queries(self, clazz, train_job_id, queries, output, logger):
        detector_inst = clazz()
        detector_inst.init()

        logger.info('upload data')
        while True:
            try:
                detector_inst.upload_queries(train_job_id, queries, output, logger)
            except:
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
