import time
import logging
import os
import traceback
import shutil

from rafiki.db import Database
from rafiki.container import DockerSwarmContainerManager
from .services_manager import ServicesManager

import pickle
import requests
import zipfile
from urllib.parse import urlparse
import random

from rafiki.constants import TaskType, DatasetProtocol

logger = logging.getLogger(__name__)

class DataRepository(object):

    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._services_manager = ServicesManager(db, container_manager)
        self._cwd = '/home/wubiao/rafiki-concept-drift'


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

    def create_new_dataset(self, train_job_id, query_index):
        #TODO: filter away data before query_index
        dataset_folder = 'dataset'
        feedback_folder = 'feedback'

        random.seed(0)

        dataset_info = {}
        if os.path.exists(os.path.join(self._cwd, train_job_id, dataset_folder, 'dataset_info.pkl')):
            dataset_info = pickle.load(open(os.path.join(self._cwd, train_job_id, dataset_folder, 'dataset_info.pkl'), 'rb'))
        else:
            if not os.path.exists(os.path.join(self._cwd, train_job_id, dataset_folder)):
                os.makedirs(os.path.join(self._cwd, train_job_id, dataset_folder))

            train_job = self._db.get_train_job(train_job_id)
            train_uri = train_job.train_dataset_uri
            test_uri = train_job.test_dataset_uri
            task = train_job.task

            if not (self._is_zip(train_uri)):
                raise Exception('{} compression not supported'.format(train_uri))

            if not (self._is_zip(test_uri)):
                raise Exception('{} compression not supported'.format(test_uri))

            if not self._is_image_classification(task) and not self.is_feature_vector_classification(task):
                raise Exception('{} task not supported'.format(task))

            parsed_train_uri = urlparse(train_uri)
            parsed_test_uri = urlparse(test_uri)
            train_protocol = '{uri.scheme}'.format(uri=parsed_train_uri)
            test_protocol = '{uri.scheme}'.format(uri=parsed_test_uri)
            
            if not (self._is_http(train_protocol) or self._is_https(train_protocol)):
                raise Exception('Dataset URI scheme not supported: {}'.format(train_protocol))

            if not (self._is_http(test_protocol) or self._is_https(test_protocol)):
                raise Exception('Dataset URI scheme not supported: {}'.format(test_protocol))

            train_file_name = os.path.basename(parsed_train_uri.path)
            test_file_name = os.path.basename(parsed_test_uri.path)
            train_folder = train_file_name.split('.')[0]
            test_folder = test_file_name.split('.')[0]

            uri_pairs = [(train_uri,train_file_name), (test_uri,test_file_name)]

            for uri,file_name in uri_pairs:
                response = requests.get(uri, stream=True)
                handle = open(os.path.join(self._cwd, train_job_id, dataset_folder, file_name), "wb")
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:  # filter out keep-alive new chunks
                        handle.write(chunk)
                handle.close()

                with zipfile.ZipFile(os.path.join(self._cwd, train_job_id, dataset_folder, file_name)) as zf:
                    zf.extractall(os.path.join(self._cwd, train_job_id, dataset_folder))

            dataset_info['version'] = 1
            dataset_info['train'] = train_folder
            dataset_info['test'] = test_folder
            dataset_info[train_folder] = {}
            dataset_info[test_folder] = {}

            data_folders = [train_folder, test_folder]

            # TODO: check the folder structure after extraction
            for folder in data_folders:
                for folder1 in os.listdir(os.path.join(self._cwd, train_job_id, dataset_folder, folder)):
                    if os.path.isdir(os.path.join(self._cwd, train_job_id, dataset_folder, folder, folder1)):
                        for file in os.listdir(os.path.join(self._cwd, train_job_id, dataset_folder, folder, folder1)):
                            if folder1 in dataset_info[folder]:
                                dataset_info[folder][folder1].append(file)
                            else:
                                dataset_info[folder][folder1] = [file]

            # sort the files in each folder according to the added-in order
            for folder in data_folders:
                for label,files in dataset_info[folder].items():
                    random.shuffle(files)

            for folder in data_folders:
                for label,files in dataset_info[folder].items():
                    print(len(files))

        train_folder = dataset_info['train']
        test_folder = dataset_info['test']
        data_folders = [train_folder, test_folder]

        feedback_info = {}

        for folder in os.listdir(os.path.join(self._cwd, train_job_id, feedback_folder)):
            if os.path.isdir(os.path.join(self._cwd, train_job_id, feedback_folder, folder)):
                if folder in dataset_info[train_folder] and folder in dataset_info[test_folder]:
                    for file in os.listdir(os.path.join(self._cwd, train_job_id, feedback_folder, folder)):
                        if folder in feedback_info:
                            feedback_info[folder].append(file)
                        else:
                            feedback_info[folder] = [file]

        for folder, files in feedback_info.items():
            files.sort(key=lambda x: int(x.split('.')[0].split('_')[1]), reverse=True)

        print(feedback_info)

        train_folder = dataset_info['train']
        test_folder = dataset_info['test']
        data_folders = [train_folder, test_folder]

        assign_files = {}
        assign_files[train_folder] = {}
        assign_files[test_folder] = {}

        for folder, files in feedback_info.items():
            train_size = len(dataset_info[train_folder][folder])
            test_size = len(dataset_info[test_folder][folder])
            for file in files:
                if random.randint(0, train_size+test_size) < train_size:
                    if folder in assign_files[train_folder]:
                        assign_files[train_folder][folder].append(file)
                    else:
                        assign_files[train_folder][folder] = [file]
                else:
                    if folder in assign_files[test_folder]:
                        assign_files[test_folder][folder].append(file)
                    else:
                        assign_files[test_folder][folder] = [file]

        print(assign_files)

        # create new dataset
        for data_folder in data_folders:
            for folder,content in assign_files[data_folder].items():
                replace_size = len(content)
                original_size = len(dataset_info[data_folder][folder])
                if replace_size > original_size:
                    replace_size = original_size
            
                for i in range(original_size-replace_size, original_size):
                    os.remove(os.path.join(self._cwd, train_job_id, dataset_folder, data_folder, folder, dataset_info[data_folder][folder][i]))
                for i in range(0, replace_size):
                    shutil.move(os.path.join(self._cwd, train_job_id, feedback_folder, folder, content[i]), os.path.join(self._cwd, train_job_id, dataset_folder, data_folder, folder, content[i]))
                tmp_list = dataset_info[data_folder][folder][:(original_size-replace_size)]
                dataset_info[data_folder][folder] = assign_files[data_folder][folder] + tmp_list

        # store dataset_info
        dataset_info['version'] += 1
        pickle.dump(dataset_info, open(os.path.join(self._cwd, train_job_id, dataset_folder, 'dataset_info.pkl'), 'wb'))

        # remove original zip
        if os.path.exists(os.path.join(self._cwd, train_job_id, dataset_folder, dataset_info['train']+'.zip')):
            os.remove(os.path.join(self._cwd, train_job_id, dataset_folder, dataset_info['train']+'.zip'))
        if os.path.exists(os.path.join(self._cwd, train_job_id, dataset_folder, dataset_info['test']+'.zip')):
            os.remove(os.path.join(self._cwd, train_job_id, dataset_folder, dataset_info['test']+'.zip'))

        original_wd = os.getcwd()
        print(original_wd)
        os.chdir(os.path.join(self._cwd, train_job_id, dataset_folder))
        zipf = zipfile.ZipFile(dataset_info['train']+'.zip', 'w', zipfile.ZIP_DEFLATED)
        self._zipdir(dataset_info['train'], zipf)
        zipf.close()

        zipf = zipfile.ZipFile(dataset_info['test']+'.zip', 'w', zipfile.ZIP_DEFLATED)
        self._zipdir(dataset_info['test'], zipf)
        zipf.close()

        os.chdir(original_wd)

        return {
            'created': True,
            'train_dataset_uri': 'http://{}:{}{}'.format('localhost', 8007, os.path.join(self._cwd, train_job_id, dataset_folder, dataset_info['train']+'.zip')),
            'test_dataset_uri': 'http://{}:{}{}'.format('localhost', 8007, os.path.join(self._cwd, train_job_id, dataset_folder, dataset_info['test']+'.zip'))
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

    def _is_zip(self, uri):
        return '.zip' in uri

    def _is_http(self, protocol):
        return protocol == DatasetProtocol.HTTP

    def _is_https(self, protocol):
        return protocol == DatasetProtocol.HTTPS

    def _is_image_classification(self, task):
        return task == TaskType.IMAGE_CLASSIFICATION

    def _is_feature_vector_classification(self, task):
        return task == TaskType.FEATURE_VECTOR_CLASSIFICATION

    def _zipdir(self, path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))