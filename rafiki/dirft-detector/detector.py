import abc

from rafiki.constants import DetectorType

class InvalidDriftDetectorTypeException(Exception):
    pass

class BaseDriftDetector(abc.ABC):
    '''
    Rafiki's base DriftDetector class
    '''   

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def detect(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def feedback(self, query_id, result):
        raise NotImplementedError()


def make_drift_detector(knob_config, detector_type=DetectorType.ACC_DD):
    if detector_type == DetectorType.ACC_DD:
        # from .btb_gp_advisor import BtbGpAdvisor
        # return BtbGpAdvisor(knob_config)
    else:
        raise InvalidDriftDetectorTypeException()