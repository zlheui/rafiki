import os
import json
import abc

from rafiki.drift_detection_method.method import BaseMethod
from rafiki.db import Database


class AccDetector(BaseMethod):
    '''
    Change detection based on feedback accuracy
    '''   

    def init(self, db=DataBase()):
        self._db = db


    def update_on_query(self):
        pass


    def update_on_feedback(self):
        raise(NotImplementedError)



