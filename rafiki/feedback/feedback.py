import time
import logging
import os
import uuid
import traceback
import pprint
import numpy as np
from rafiki.db import Database

logger = logging.getLogger(__name__)


class Feedback(object):

    def __init__(self, db=Database()):
        self._db = db


    def create_feedback(self, query_index, label=None):
        is_added = False
        label_is_none = True

        if label is not None:
            self._db.connect()
            feedback = self._db.create_feedback(query_index=int(query_index), label=label)
            self._db.commit()
            label_is_none = False
            is_added = True
        
        return {'query_index': query_index, 'is_added': is_added, 'label_is_none': label_is_none} 
