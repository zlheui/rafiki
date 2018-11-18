import time
import logging
import os
import traceback
import shutil

from rafiki.db import Database
from rafiki.container import DockerSwarmContainerManager
from .services_manager import ServicesManager

logger = logging.getLogger(__name__)

class DataRepository(object):

    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._services_manager = ServicesManager(db, container_manager)
        self._cwd = '/home/zhulei/rafiki-concept-drift'


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

    def remove_train_job_folder(self, train_job_id):
        shutil.rmtree(os.path.join(self._cwd, train_job_id), ignore_errors=True)

        return {
            'removed': True
        }

    def remove_all_folders(self):
        shutil.rmtree(self._cwd, ignore_errors=True)

        return {
            'removed': True
        }

    def print_folder_structure(self, train_job_id):
        folder_structure = self.list_files(os.path.join(self._cwd, train_job_id))

        return {
            'folder_structure': folder_structure
        }

    def create_retrain_service(self, train_job_id, query_index):
        service = self._services_manager.create_retrain_service(train_job_id, query_index)

        return {
            'created': True,
            'id': service.id
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