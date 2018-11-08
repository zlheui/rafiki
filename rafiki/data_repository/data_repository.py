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
        self._cwd = '/'


    def list_files(self, startpath):
        output = ''
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            output += '{}{}/'.format(indent, os.path.basename(root)) + '\n'
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                output += '{}{}'.format(subindent, f) + '\n'
        return output

    def print_folder_structure(self):
        query_structure = self.list_files(os.path.join(self._cwd, 'query'))
        feedback_structure = self.list_files(os.path.join(self._cwd, 'feedback'))
        dataset_structure = self.list_files(os.path.join(self._cwd, 'dataset'))

        return {
            'query_structure': query_structure,
            'feedback_structure': feedback_structure,
            'dataset_structure': dataset_structure
        }

    def create_new_dataset(self, train_job_id, query_index):
        files = [e for e in os.listdir(os.path.join(self._cwd, 'dataset'))]

        if len(files) == 0:
            # TODO: create train_URI and test_URI
            pass

        return {
            'created': True,
            'train_dataset_uri': 'abc',
            'test_dataset_uri': 'abc'
        }

    def create_data_repository_service(self, service_type):
        service = self._services_manager.create_data_repository_service(service_type)

        return {
            'id': service.id
        }

    def stop_data_repository_service(self, service_type):
        service_id = self._services_manager.stop_data_repository_service(service_type)

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