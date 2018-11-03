import time
import logging
import os
import traceback
import pprint
from rafiki.db import Database
from rafiki.constants import TaskType
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

logger = logging.getLogger(__name__)


class DataRepository(object):

    def __init__(self, db=Database()):
        self._cwd = '/'
        self._db = db

    def create_new_dataset(self, train_job_id):
        output = [e for e in os.listdir(os.path.join(self._cwd, train_job_id, 'query'))]

        return {
            'queries': output
        }

    def create_query(self, train_job_id, data_point=None):
        is_added = False
        query_index = -1
        if data_point is not None:
            
            self._db.connect()
            train_job = self._db.get_train_job(train_job_id)
            query_stats = self._db.get_query_stats_by_train_job_id(train_job_id)
            if query_stats is None:
                query_stats = self._db.create_query_stats(train_job_id)
                self._db.commit()

            query_index = query_stats.next_query_index
            self._create_query_folder(train_job_id)
            if train_job.task == TaskType.IMAGE_CLASSIFICATION:
                if len(np.array(data_point).shape) == 2:
                    plt.imsave(os.path.join(self._cwd, train_job_id, 'query', str(query_index)+'.png'), np.array(data_point), cmap=cm.gray)
                else:
                    plt.imsave(os.path.join(self._cwd, train_job_id, 'query', str(query_index)+'.png'), np.array(data_point))
                is_added = True
            elif train_job.task == TaskType.FEATURE_VECTOR_CLASSIFICATION:
                with open(os.path.join(self._cwd, train_job_id, 'query', str(query_index)+'.csv'), 'w') as f:
                    f.write(','.join([str(e) for e in data_point]))
                is_added = True
            else:
                raise NotImplementedError

            if is_added:
                self._db.update_query_stats(query_stats, query_index+1)
                self._db.commit()
        return {
            'is_added': is_added,
            'query_index': query_index
        }


    def _create_query_folder(self, train_job_id):
        if not os.path.exists(os.path.join(self._cwd, train_job_id)):
            os.makedirs(os.path.join(self._cwd, train_job_id))
        if not os.path.exists(os.path.join(self._cwd, train_job_id, 'query')):
            os.makedirs(os.path.join(self._cwd, train_job_id, 'query'))
