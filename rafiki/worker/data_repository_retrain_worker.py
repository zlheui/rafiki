import time
import logging
import os
import traceback
import pprint

from rafiki.constants import ServiceType, TaskType, Prefixes
from rafiki.db import Database
from rafiki.client import Client

import pickle
import requests
import zipfile
from urllib.parse import urlparse
import random
import shutil

from rafiki.constants import TaskType, DatasetProtocol
from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD

logger = logging.getLogger(__name__)

class DataRepositoryRetrainWorker(object):
    def __init__(self, service_id, train_job_id, query_index, db=Database()):
        self._service_id = service_id
        self._cwd = os.environ['CONCEPT_DRIFT_FOLDER']
        self._train_job_id = train_job_id
        self._query_index = query_index
        self._dataset_folder = 'dataset'
        self._feedback_folder = 'feedback'
        self._db = db
        self._db.connect()
        self._db_connected = False
        self._client = None


    def start(self):
        logger.info('Starting data repository retrain worker for service of id {}...' \
            .format(self._service_id))
        
        logger.info('Create new dataset')
        uris = self.create_new_dataset()
        logger.info('Finish creating new dataset')

        logger.info('Create new train job')
        old_train_job = self._db.get_train_job(self._train_job_id)
        if self._client == None:
            self._client = self._make_client()

        self._client.create_train_job(
           app=old_train_job.app,
           task=old_train_job.task,
           train_dataset_uri=uris['train_dataset_uri'],
           test_dataset_uri=uris['test_dataset_uri'],
           budget_type=old_train_job.budget_type,
           budget_amount=old_train_job.budget_amount
        )

        #wait until train job completed
        while True:
           time.sleep(10)
           try:
               train_job = self._client.get_train_job(app=old_train_job.app)
               if train_job.get('status') == 'COMPLETED':
                   break
           except:
               pass

        logger.info('Finish creating new train job')

        logger.info('Subscribe new train job best trials')
        trials = self._db.get_best_trials_of_train_job(train_job['id'])
        detectors = self._db.get_all_detectors()
        for trial in trials:
            for detector in detectors:
                self._client.subscribe_drift_detection_service(trial.id, detector.name)
        self._db.commit()

        logger.info('Finish subscribing best trials')


        #stop old inference job
        self._client.stop_inference_job(app=old_train_job.app, app_version=old_train_job.app_version)

        # wait for a while to make sure no queries sent to predictor
        time.sleep(5)

        logger.info('Update new train job next query index')
        old_query_stats = self._db.get_query_stats_by_train_job_id(old_train_job.id)
        query_stats = self._db.get_query_stats_by_train_job_id(train_job['id'])
        if query_stats is None:
            query_stats = self._db.create_query_stats(train_job['id'])
            self._db.commit()
        self._db.update_query_stats(query_stats, old_query_stats.next_query_index)
        self._db.commit()

        logger.info('Finish updating new train job next query index')

        #deploy new inference job
        self._client.create_inference_job(app=train_job['app'], app_version=train_job['app_version'])


    def create_new_dataset(self):
        #get train job and check task type
        train_job = self._db.get_train_job(self._train_job_id)
        task = train_job.task
        if (not self._is_image_classification(task)) and (not self._is_feature_vector_classification(task)):
            raise Exception('{} task not supported'.format(task))
        
        random.seed(0)
        #unified data file name, for single csv file data sets
        DATA_FILE_NAME = 'data.csv'

        dataset_info = {}
        if os.path.exists(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, 'dataset_info.pkl')):
            dataset_info = pickle.load(open(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, 'dataset_info.pkl'), 'rb'))
        else:
            if not os.path.exists(os.path.join(self._cwd, self._train_job_id, self._dataset_folder)):
                os.makedirs(os.path.join(self._cwd, self._train_job_id, self._dataset_folder))

            train_uri = train_job.train_dataset_uri
            test_uri = train_job.test_dataset_uri

            if not (self._is_zip(train_uri)):
                raise Exception('{} compression not supported'.format(train_uri))

            if not (self._is_zip(test_uri)):
                raise Exception('{} compression not supported'.format(test_uri))

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
                handle = open(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, file_name), "wb")
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:  # filter out keep-alive new chunks
                        handle.write(chunk)
                handle.close()

                with zipfile.ZipFile(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, file_name)) as zf:
                    zf.extractall(os.path.join(self._cwd, self._train_job_id, self._dataset_folder))

            dataset_info['version'] = 1
            dataset_info['train'] = train_folder
            dataset_info['test'] = test_folder
            dataset_info[train_folder] = {}
            dataset_info[test_folder] = {}

            data_folders = [train_folder, test_folder]

            # TODO: check the folder structure after extraction, this is different per task
            if self._is_image_classification(task):
                for folder in data_folders:
                    for folder1 in os.listdir(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, folder)):
                        if os.path.isdir(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, folder, folder1)):
                            for file in os.listdir(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, folder, folder1)):
                                if folder1 in dataset_info[folder]:
                                    dataset_info[folder][folder1].append(file)
                                else:
                                    dataset_info[folder][folder1] = [file]

                # shuffle
                for folder in data_folders:
                    for label,files in dataset_info[folder].items():
                        random.shuffle(files)
                        
            elif self._is_feature_vector_classification(task):
                for folder in data_folders:
                    #merge files first
                    all_content = []
                    for file in oslistdir(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, folder)):
                        #merge files first
                        content = np.genfromtxt(file, delimiter=',')
                        all_content = all_content + [content]
                        os.remove(file)
                    #save data file
                    numpy.savetxt(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, folder, DATA_FILE_NAME),\
                                  all_content, delimiter=",")
                    
                    #record entry row index
                    for index in range(len(allContent)):
                        folder1 = allContent[index][-1]
                        if folder1 in dataset_info[folder]:
                            dataset_info[folder][folder1].append(index)
                        else:
                            dataset_info[folder][folder1] = [index]

                    # shuffle
                    for folder in data_folders:
                        for folder, files in dataset_info[folder].items():
                            random.shuffle(files)

        train_folder = dataset_info['train']
        test_folder = dataset_info['test']
        data_folders = [train_folder, test_folder]

        feedback_info = {}

        for folder in os.listdir(os.path.join(self._cwd, self._train_job_id, self._feedback_folder)):
            if os.path.isdir(os.path.join(self._cwd, self._train_job_id, self._feedback_folder, folder)):
                if folder in dataset_info[train_folder] and folder in dataset_info[test_folder]:
                    for file in os.listdir(os.path.join(self._cwd, self._train_job_id, self._feedback_folder, folder)):
                        # only add files until the query_index
                        file_index = int(file.split('.')[0].split('_')[1])
                        if file_index < int(self._query_index):
                            if folder in feedback_info:
                                feedback_info[folder].append(file)
                            else:
                                feedback_info[folder] = [file]

        for folder, files in feedback_info.items():
            files.sort(key=lambda x: int(x.split('.')[0].split('_')[1]), reverse=True)

        print(feedback_info)

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
            if self._is_image_classification(task):
                for folder, content in assign_files[data_folder].items():
                    replace_size = len(content)
                    original_size = len(dataset_info[data_folder][folder])
                    if replace_size > original_size:
                        replace_size = original_size

                    for i in range(original_size-replace_size, original_size):
                        os.remove(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, \
                                               data_folder, folder, dataset_info[data_folder][folder][i]))
                    for i in range(0, replace_size):
                        shutil.move(os.path.join(self._cwd, self._train_job_id, self._feedback_folder, folder, content[i]), \
                                    os.path.join(self._cwd, self._train_job_id, self._dataset_folder, data_folder, folder, content[i]))
                    tmp_list = dataset_info[data_folder][folder][:(original_size-replace_size)]
                    dataset_info[data_folder][folder] = assign_files[data_folder][folder] + tmp_list
            elif self._is_feature_vector_classification(task):
                #load content from the only data file
                all_content = np.genfromtxt(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, \
                                                         data_folder, DATA_FILE_NAME), delimiter=',')
                for folder, content in assign_files[data_folder].items():
                    replace_size = len(content)
                    original_size = len(dataset_info[data_folder][folder])
                    if replace_size > original_size:
                        replace_size = original_size

                    for i in range(original_size-replace_size, original_size):
                        os.remove(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, \
                                               data_folder, folder, dataset_info[data_folder][folder][i]))
                    for i in range(0, replace_size):
                        content = np.genfromtxt(os.path.join(self._cwd, self._train_job_id, self._feedback_folder, folder, content[i]), \
                                                delimiter=',')
                        #replace data in ith row with feedback data
                        all_content[dataset_info[data_folder][folder][i]] = content[1][0]
                    #rotate the indexing
                    dataset_info[data_folder][folder] = dataset_info[data_folder][folder][replace_size:] + \
                                                        dataset_info[data_folder][folder][:replace_size]
                #save data file
                numpy.savetxt(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, data_folder, DATA_FILE_NAME),\
                                  all_content, delimiter=",")
                    

        # store dataset_info
        dataset_info['version'] += 1
        pickle.dump(dataset_info, open(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, 'dataset_info.pkl'), 'wb'))

        # remove original zip
        if os.path.exists(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, dataset_info['train']+'.zip')):
            os.remove(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, dataset_info['train']+'.zip'))
        if os.path.exists(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, dataset_info['test']+'.zip')):
            os.remove(os.path.join(self._cwd, self._train_job_id, self._dataset_folder, dataset_info['test']+'.zip'))

        original_wd = os.getcwd()
        print(original_wd)
        os.chdir(os.path.join(self._cwd, self._train_job_id, self._dataset_folder))
        zipf = zipfile.ZipFile(dataset_info['train']+'.zip', 'w', zipfile.ZIP_DEFLATED)
        self._zipdir(dataset_info['train'], zipf)
        zipf.close()

        zipf = zipfile.ZipFile(dataset_info['test']+'.zip', 'w', zipfile.ZIP_DEFLATED)
        self._zipdir(dataset_info['test'], zipf)
        zipf.close()

        os.chdir(original_wd)

        return {
            'train_dataset_uri': 'http://{}:{}{}'.format(os.environ['DATA_REPOSITORY_HOST'], os.environ['DATA_REPOSITORY_PORT'], \
                os.path.join(self._cwd, self._train_job_id, self._dataset_folder, dataset_info['train']+'.zip')),
            'test_dataset_uri': 'http://{}:{}{}'.format(os.environ['DATA_REPOSITORY_HOST'], os.environ['DATA_REPOSITORY_PORT'], \
                os.path.join(self._cwd, self._train_job_id, self._dataset_folder, dataset_info['test']+'.zip'))
        }

    def stop(self):
        logger.info('Stopping data repository retrain worker for service of id {}...' \
            .format(self._service_id))
        self._db.disconnect()

    def _make_client(self):
        admin_host = os.environ['ADMIN_HOST']
        admin_port = os.environ['ADMIN_PORT']
        advisor_host = os.environ['ADVISOR_HOST']
        advisor_port = os.environ['ADVISOR_PORT']
        drift_detector_host = os.environ['DRIFT_DETECTOR_HOST']
        drift_detector_port = os.environ['DRIFT_DETECTOR_PORT']
        superadmin_email = SUPERADMIN_EMAIL
        superadmin_password = SUPERADMIN_PASSWORD
        client = Client(admin_host=admin_host, 
                        admin_port=admin_port, 
                        advisor_host=advisor_host,
                        advisor_port=advisor_port,
                        drift_detector_host=drift_detector_host,
                        drift_detector_port=drift_detector_port)
        client.login(email=superadmin_email, password=superadmin_password)
        return client

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
