import sys
import os
import signal
import traceback
import logging

from rafiki.constants import ServiceType
from rafiki.utils.log import configure_logging

service_id = os.environ['RAFIKI_SERVICE_ID']
service_type = os.environ['RAFIKI_SERVICE_TYPE']
container_id = os.environ.get('HOSTNAME', 'localhost')

configure_logging('service-{}-{}'.format(service_id, container_id))

logger = logging.getLogger(__name__)

def sigterm_handler(_signo, _stack_frame):
    print("SIGTERM received: %s, %s" % (_signo, _stack_frame))
    exit_worker()

worker = None
def exit_worker():
    if worker is not None:
        worker.stop()
        print('Worker stopped gracefully.')  

signal.signal(signal.SIGTERM, sigterm_handler)

try:
    if service_type == ServiceType.REPOSITORY_QUERY:
        from rafiki.worker.data_repository_query_worker import DataRepositoryQueryWorker
        worker = DataRepositoryQueryWorker(service_id)
        worker.start()
    elif service_type == ServiceType.REPOSITORY_FEEDBACK:
        from rafiki.worker.data_repository_feedback_worker import DataRepositoryFeedbackWorker
        worker = DataRepositoryFeedbackWorker(service_id)
        worker.start()
    elif service_type == ServiceType.REPOSITORY_RETRAIN:
        from rafiki.worker.data_repository_retrain_worker import DataRepositoryRetrainWorker
        train_job_id = os.environ['TRAIN_JOB_ID']
        query_index = os.environ['QUERY_INDEX']
        worker = DataRepositoryRetrainWorker(service_id, train_job_id, query_index)
        worker.start()
    else:
        raise Exception('Invalid service type: {}'.format(service_type))
    exit_worker()
except Exception as e:
    logger.error('Error while running worker:')
    logger.error(traceback.format_exc())
    exit_worker()
    raise e