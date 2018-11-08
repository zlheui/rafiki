import os
import json
import abc

from rafiki.drift_detection_method.method import BaseMethod
from rafiki.db import Database


class AccDetector(BaseMethod):
    '''
    Change detection based on feedback accuracy
    '''

    def update_on_queries(self, train_job_id, queries):
        return False, None


    def update_on_feedbacks(self, train_job_id, feedbacks):
        return False, None



