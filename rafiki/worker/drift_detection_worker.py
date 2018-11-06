import time
import logging
import os
import traceback
import pprint

from rafiki.utils.model import load_detector_class
from rafiki.db import Database
from rafiki.cache import Cache
from rafiki.config import DRIFT_WORKER_SLEEP, DRIFT_DETECTION_BATCH_SIZE

class DriftDetectionWorker(Object):
    def __init__(self, service_id, cache=Cache(), db=Database(isolation_level='REPEATABLE_READ')):
        self._cache = cache
        self._db = db
        self._service_id = service_id
        self._detectors = {}

    def start(self):
        logger.info('Starting drift detection worker for service of id {}...' \
            .format(self._service_id))

        # Add to set of running workers
        self._cache.add_drift_detection_worker(self._service_id)

        while True:
            (query_ids, queries) = \
                self._cache.pop_queries_of_worker(self._service_id, DRIFT_DETECTION_BATCH_SIZE)

            if len(queries) > 0:
                logger.info('Detecting concept drift for queries...')
                logger.info(queries)

                
                
                
                predictions = None
                try:
                    predictions = self._model.predict(queries)
                    predictions = [parse_model_prediction(x) for x in predictions]
                except Exception:
                    logger.error('Error while making predictions:')
                    logger.error(traceback.format_exc())
                    
                if predictions is not None:
                    logger.info('Predictions:')
                    logger.info(predictions)

                    for (query_id, prediction) in zip(query_ids, predictions):
                        self._cache.add_prediction_of_worker(self._service_id, query_id, prediction)
        
        time.sleep(DRIFT_WORKER_SLEEP)