import time
import logging
import os
import uuid
import traceback
import pprint
from rafiki.cache import Cache
from rafiki.db import Database
from rafiki.constants import ServiceType

logger = logging.getLogger(__name__)


class Feedback(object):

    def __init__(self, db=Database(), cache=Cache()):
        self._db = db
        self._cache = cache

    def create_feedback(self, train_job_id, query_index, label=None):
        logger.info('Received feedback:')
        logger.info(train_job_id + '_' + query_index + '_' + label)

        is_added = False
        label_is_none = True

        if label is not None:
            self._db.connect()
            feedback = self._db.create_feedback(query_index=int(query_index), label=label)
            self._db.commit()
            label_is_none = False
            is_added = True

            train_job = self._db.get_train_job(train_job_id)
            if train_job.subscribe_to_drift_detection_service:
                # send feedback to drift detector
                running_drift_detection_worker_ids = self._cache.get_drift_detection_workers(ServiceType.DRIFT_FEEDBACK)
                if len(running_drift_detection_worker_ids) > 0:
                    con_drift_feedback_id = self._cache.add_feedback_of_worker(running_drift_detection_worker_ids[0], train_job_id, feedback.id, query_index, label)
                # send feedback to data repository
                running_data_repository_worker_ids = self._cache.get_data_repository_workers(ServiceType.REPOSITORY_FEEDBACK)
                if len(running_data_repository_worker_ids) > 0:
                    data_repo_feedback_id = self._cache.add_feedback_of_worker(running_data_repository_worker_ids[0], train_job_id, feedback.id, query_index, label)
            self._db.disconnect()

        return {'query_index': query_index, 'is_added': is_added, 'label_is_none': label_is_none} 
