import time
import logging
import os
import traceback
import pprint

from rafiki.cache import Cache
from rafiki.config import DATA_REPOSITORY_SLEEP, DATA_REPOSITORY_BATCH_SIZE
from rafiki.constants import ServiceType


class DataRepositoryWorker(Object):
    def __init__(self, service_id, cache=Cache()):
        self._cache = cache
        self._service_id = service_id

    def start(self):
        logger.info('Starting data repository worker for service of id {}...' \
            .format(self._service_id))

        # Add to set of running workers
        self._cache.add_data_repository_worker(self._service_id, ServiceType.DATA_REPOSITORY)

        while True:
            
    
    def stop(self):
        # Remove from set of running workers
        self._cache.delete_data_repository_worker(self._service_id, ServiceType.DATA_REPOSITORY)
