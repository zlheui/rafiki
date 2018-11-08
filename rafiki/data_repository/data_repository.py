import time
import logging
import os
import traceback

from rafiki.db import Database
from rafiki.container import DockerSwarmContainerManager
from .services_manager import ServicesManager

logger = logging.getLogger(__name__)

class DataRepository(object):

    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._services_manager = ServicesManager(db, container_manager)

    def create_new_dataset(self, train_job_id):
        output = [e for e in os.listdir(os.path.join(self._cwd, train_job_id, 'query'))]

        return {
            'queries': output
        }

    def create_data_repository_service(self):
        service = self._services_manager.create_data_repository_service()

        return {
            'id': service.id
        }

    def stop_data_repository_service(self):
        service_id = self._services_manager.stop_data_repository_service()

        return {
            'id': service_id
        }

    def __enter__(self):
        self.connect()

    def connect(self):
        self._db.connect()

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()

    def disconnect(self):
        self._db.disconnect()