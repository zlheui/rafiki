import time
import logging
import os
import traceback
import pprint

from rafiki.utils.model import load_detector_class
from rafiki.db import Database
from rafiki.cache import Cache

class DriftDetectionWorker(Object):
    def __init__(self, service_id, cache=Cache(), db=Database()):
        self._cache = cache
        self._db = db
        self._service_id = service_id
        self._detectors = {}

    def start(self):
        logger.info('Starting drift detection worker for service of id {}...' \
            .format(self._service_id))

        query_index = 0
        while True:
            query = self._cache.pop_prediction_of_worker(worker_id, query_id)