import time
import logging
import os
import traceback
import pprint

from rafiki.cache import Cache
from rafiki.db import Database
from rafiki.config import DATA_REPOSITORY_SLEEP, DATA_REPOSITORY_BATCH_SIZE
from rafiki.constants import ServiceType, TaskType

import shutil

class DataRepositoryFeedbackWorker(object):
    def __init__(self, service_id, cache=Cache(), db=Database()):
        self._cache = cache
        self._db = db
        self._service_id = service_id
        self._cwd = '/'

    def start(self):
        logger.info('Starting data repository worker for service of id {}...' \
            .format(self._service_id))

        # Add to set of running workers
        self._cache.add_data_repository_worker(self._service_id, ServiceType.REPOSITORY_FEEDBACK)

        while True:
            (feedback_ids, train_job_ids, query_indexes, labels) = \
                self._cache.pop_feedbacks_of_worker(self._service_id, DRIFT_DETECTION_BATCH_SIZE)

            if len(labels) > 0:
                
                self._db.connect()
                for (train_job_id, query_index, label) in zip(train_job_ids, query_indexes, labels):
                    # create feedback folder
                    self._create_feedback_folder(train_job_id)
                    if not os.path.exists(os.path.join(self._cwd, train_job_id, 'feedback', label)):
                        os.makedirs(os.path.join(self._cwd, train_job_id, 'feedback', label))
                    
                    # move unlabeled data to feedback folder
                    train_job = self._db.get_train_job(train_job_id)
                    if train_job.task == TaskType.IMAGE_CLASSIFICATION:
                        shutil.move(os.path.join(self._cwd, train_job_id, 'query', str(query_index)+'.png'), \
                         os.path.join(self._cwd, train_job_id, 'feedback', label, str(query_index)+'.png'))
                    elif train_job.task == TaskType.FEATURE_VECTOR_CLASSIFICATION:
                        shutil.move(os.path.join(self._cwd, train_job_id, 'query', str(query_index)+'.csv'), \
                         os.path.join(self._cwd, train_job_id, 'feedback', label, str(query_index)+'.csv'))
                    else:
                        raise NotImplementedError

            time.sleep(DATA_REPOSITORY_SLEEP)


    def stop(self):
        # Remove from set of running workers
        self._cache.delete_data_repository_worker(self._service_id, ServiceType.REPOSITORY_FEEDBACK)

    def _create_feedback_folder(self, train_job_id):
        if not os.path.exists(os.path.join(self._cwd, train_job_id)):
            os.makedirs(os.path.join(self._cwd, train_job_id))
        if not os.path.exists(os.path.join(self._cwd, train_job_id, 'feedback')):
            os.makedirs(os.path.join(self._cwd, train_job_id, 'feedback'))