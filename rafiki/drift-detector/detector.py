import abc

from rafiki.constants import DetectorType
from rafiki.db import Database

class InvalidDriftDetectorTypeException(Exception):
    pass

class BaseDriftDetector(abc.ABC):
    '''
    Rafiki's base DriftDetector class
    '''   

    def __init__(self, db=Database()):
        self._db = db

    @abc.abstractmethod
    def detect(self):
        raise NotImplementedError()

    def feedback(self, query_id, label):
        self._db.connect()
        


def make_drift_detector(knob_config, detector_type=DetectorType.ACC_DD):
    if detector_type == DetectorType.ACC_DD:
        # from .btb_gp_advisor import BtbGpAdvisor
        # return BtbGpAdvisor(knob_config)
    else:
        raise InvalidDriftDetectorTypeException()