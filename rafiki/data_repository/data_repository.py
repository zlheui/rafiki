import time
import logging
import os
import traceback
import pprint

logger = logging.getLogger(__name__)


class DataRepository(object):

    def __init__(self):
        self._cwd = '/'

    def create_new_dataset(self, train_job_id):
        raise NotImplementedError

    
