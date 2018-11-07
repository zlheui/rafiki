import os
import json
import abc

from rafiki.drift_detection_method.method import BaseMethod
from rafiki.db import Database


class AccDetector(BaseMethod):
    '''
    Change detection based on feedback accuracy
    '''

    def update_on_queries(self):
        pass


    def update_on_feedbacks(self):
        raise(NotImplementedError)



