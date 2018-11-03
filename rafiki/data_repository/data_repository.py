import time
import logging
import os
import traceback
import pprint
from rafiki.db import Database
from rafiki.constants import TaskType

logger = logging.getLogger(__name__)


class DataRepository(object):

    def __init__(self, db=Database()):
        self._cwd = '/'
        self._db = db

    def create_new_dataset(self, train_job_id):
        raise NotImplementedError

    def create_query(self, train_job_id, data_point=None):
    	self._db.connect()
    	train_job = self._db.get_train_job(train_job_id)
    	if train_job.task == TaskType.IMAGE_CLASSIFICATION:
    		query_stats = self._db.get_query_stats(train_job_id)
    	else:
    		raise NotImplementedError
