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
    def feedback(self, query_id):
        raise NotImplementedError()


def make_DriftDetector(knob_config, advisor_type=AdvisorType.BTB_GP):
    if advisor_type == AdvisorType.BTB_GP:
        from .btb_gp_advisor import BtbGpAdvisor
        return BtbGpAdvisor(knob_config)
    else:
        raise InvalidAdvisorTypeException()