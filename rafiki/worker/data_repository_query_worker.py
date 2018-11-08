import time
import logging
import os
import traceback
import pprint

from rafiki.db import Database
from rafiki.cache import Cache
from rafiki.config import DATA_REPOSITORY_SLEEP, DATA_REPOSITORY_BATCH_SIZE
from rafiki.constants import ServiceType, TaskType

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class DataRepositoryQueryWorker(object):
    def __init__(self, service_id, cache=Cache(), db=Database()):
        self._cache = cache
        self._db = db
        self._service_id = service_id
        self._cwd = '/'

    def start(self):
        logger.info('Starting data repository worker for service of id {}...' \
            .format(self._service_id))

        # Add to set of running workers
        self._cache.add_data_repository_worker(self._service_id, ServiceType.REPOSITORY_QUERY)

        while True:
            (train_job_ids, query_indexes, queries) = \
                self._cache.pop_queries_of_data_repository_worker(self._service_id, DATA_REPOSITORY_BATCH_SIZE)

            if len(train_job_ids) > 0:
                self._db.connect()
                for query_index, train_job_id in zip(query_indexes, train_job_ids):
                    train_job = self._db.get_train_job(train_job_id)
                    self._create_query_folder(train_job_id)
                    if train_job.task == TaskType.IMAGE_CLASSIFICATION:
                        tmp_index = int(query_index)
                        for query in queries:
                            if len(np.array(query).shape) == 2:
                                plt.imsave(os.path.join(self._cwd, train_job_id, 'query', str(tmp_index)+'.png'), np.array(query), cmap=cm.gray)
                            else:
                                plt.imsave(os.path.join(self._cwd, train_job_id, 'query', str(tmp_index)+'.png'), np.array(query))
                            tmp_index += 1
                    elif train_job.task == TaskType.FEATURE_VECTOR_CLASSIFICATION:
                        tmp_index = int(query_index)
                        for query in queries:
                            with open(os.path.join(self._cwd, train_job_id, 'query', str(tmp_index)+'.csv'), 'w') as f:
                                f.write(','.join([str(e) for e in query]))
                            tmp_index += 1
                    else:
                        raise NotImplementedError

            time.sleep(DATA_REPOSITORY_SLEEP)

    def stop(self):
        # Remove from set of running workers
        self._cache.delete_data_repository_worker(self._service_id, ServiceType.REPOSITORY_QUERY)

    def _create_query_folder(self, train_job_id):
        if not os.path.exists(os.path.join(self._cwd, train_job_id)):
            os.makedirs(os.path.join(self._cwd, train_job_id))
        if not os.path.exists(os.path.join(self._cwd, train_job_id, 'query')):
            os.makedirs(os.path.join(self._cwd, train_job_id, 'query'))
