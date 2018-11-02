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


    def create_feedback(self, query_id, label=None):
    	if label is not None:


