import json
import pickle
import os
import base64
import numpy as np

from rafiki.cache import Cache
from rafiki.constants import ServiceType
from rafiki.db import Database
from rafiki.drift_detection_method import BaseMethod, load_dataset_training
from rafiki.utils.model import load_model_class


class EWMADetector(BaseMethod):
    '''
    Implements a EWMA drift detection
    '''
    def get_init_param(self):
        return {
            'status': 'uninitialized',    #ready, detected
            'lamda': 6,                   #control limit
            'L': 1,                       #penalty
            'step_size': 10,              #step size
            'ncol': 1,                    #number of columns
            'nclass': 2,                  #number of classes, from 0 to nclass-1
            'feature': {},                #'mean', 'stdev', 'shigh', 'slow'
            'label': {},                  #'mean', 'stdev', 'shigh', 'slow'
            'trend': None,
            'type': None,
            'index': None,
            'query_index': None
        }

    def init(self, service_type=None, detector_name=None, job_id=None, model = None, cache=Cache(), db=Database(isolation_level='REPEATABLE_READ'), logger = None):
        self._cache = cache
        self._db = db
        self._detector_name = detector_name
        self._param = self.get_init_param()

        self._db.connect()
        #load method parameter from database 
        #otherwise taking job testing dataset
        if service_type and detector_name and job_id is not None:
            #if service_type is None, means no need to load param (currently only invoked during query upload)
            if service_type == ServiceType.DRIFT_QUERY:
            #update feature status
                param_str = self._db.get_train_job_detector_param(job_id, detector_name)
                if param_str is not None:
                    #load exisitng parma in dabase
                    self.load_parameters(param_str)
                else:
                    #training the model using testing data set in training job without the need of prediction
                    train_job = self._db.get_train_job(job_id)
                    task = train_job.task
                    uri = train_job.test_dataset_uri
                    (X, y) = self._load_dataset_training(uri, task)
                    
                    self._param['feature']['mean'] = np.mean(X, axis = 0).tolist()
                    self._param['feature']['stdev'] = np.std(X, axis = 0).tolist()
                    self._param['feature']['chigh'] = [0] * self._param['ncol']
                    self._param['feature']['clow'] = [0] * self._param['ncol']

                    #stats
                    self._param['feature']['ewma'] = [0] * self._param['ncol']
                    self._param['feature']['counter'] = len(X)

                    #trained model ready for detection
                    self._param['status'] = 'ready'
            elif service_type == ServiceType.DRIFT_FEEDBACK:
            #update label status
                param_str = self._db.get_trial_detector_param(job_id, detector_name)
                if param_str is not None:
                    #load exisitng parma in dabase
                    self.load_parameters(param_str)
                else:
                    #training the model using testing data set in training job, and the prediciton results
                    trial = self._db.get_trial(job_id)
                    train_job = self._db.get_train_job(trial.train_job_id)
                    task = train_job.task
                    uri = train_job.test_dataset_uri
                    model = self._load_model(job_id)
                    (X, y) = self._load_dataset_training(uri, task, model, logger)
                    
                    #baseline feedback stats
                    self._param['label']['mean'] = [np.mean(y[i]) for i in range(self._param['nclass'])]
                    self._param['label']['stdev'] = [np.std(y[i]) for i in range(self._param['nclass'])]
                    self._param['label']['chigh'] = [0] * self._param['nclass']
                    self._param['label']['clow'] = [0] * self._param['nclass']

                    #stats
                    self._param['label']['ewma'] = [0] * self._param['nclass']
                    self._param['label']['counter'] = len(X)

                    #trained model ready for detection
                    self._param['status'] = 'ready'
            else:
                logger.info('unsupported service type: {}'.format(service_type))
        self._db.disconnect()
    
    def update_on_queries(self, train_job_id, queries, query_index, logger):
        if self._param['status'] != 'ready':
            logger.info('model status has to be ready but is {0}'.format(self._param['status']))
            if self._param['status'] == 'drifted':
                return True, self._param['query_index']
            else:
                return False, None
        X=[row[1][0] for row in queries]
        ewma = self._param['feature']['ewma']
        mean = self._param['feature']['mean']
        stdev = self._param['feature']['stdev']
        chigh = self._param['feature']['chigh']
        clow = self._param['feature']['clow']
        lamda = self._param['lamda']
        L = self._param['L']
        step_size = self._param['step_size']

        for i in range(self._param['ncol']):
            count = self._param['feature']['counter']
            for j in range(len(X)):
                count += 1
                rang = L * stdev[i] * np.sqrt(lamda / (2 - lamda) * (1 - np.power(1 - lamda, 2 * count)))
                hlimit = mean[i] + rang
                llimit = mean[i] - rang
                ewma[i] = lamda * X[j][i] + (1 - lamda) * ewma[i]
                if ewma[i] > hlimit:
                    chigh[i] += 1
                else:
                    chigh[i] = 0
                if  ewma[i] < llimit:
                    clow[i] += 1
                else:
                    clow[i] = 0

                if (chigh[i] >= step_size and ewma[i] > hlimit) \
                     or (clow[i] >= step_size and ewma[i] < llimit):
                    self._param['status'] = 'drifted'
                    if ewma[i] > hlimit:
                        self._param['trend'] = 'up'
                    else:
                        self._param['trend'] = 'down'
                    self._param['type'] = 'feature'
                    self._param['index'] = i
                    self._param['query_index'] = query_index + j
                    return True, self._param['query_index']
        self._param['feature']['counter'] += len(X)
        return False, None
    
    def update_on_feedbacks(self, trial_id, feedbacks, logger=None):
        if self._param['status'] != 'ready':
            logger.info('model status has to be ready but is {0}'.format(self._param['status']))
            if self._param['status'] == 'drifted':
                return True, self._param['query_index']
            else:
                return False, None

        y = self._load_prediction_from_feedbacks(self._param['nclass'], trial_id, feedbacks)
        ewma = self._param['label']['ewma']
        mean = self._param['label']['mean']
        stdev = self._param['label']['stdev']
        chigh = self._param['label']['chigh']
        clow = self._param['label']['clow']
        lamda = self._param['lamda']
        L = self._param['L']
        step_size = self._param['step_size']
        
        for i in range(self._param['nclass']):
            count = self._param['label']['counter']
            for j in range(len(y[i])):
                count += 1
                rang = L * stdev[i] * np.sqrt(lamda / (2 - lamda) * (1 - np.power(1 - lamda, 2 * count)))
                hlimit = mean[i] + rang
                llimit = mean[i] - rang
                ewma[i] = lamda * y[i][j] + (1 - lamda) * ewma[i]
                if ewma[i] > hlimit:
                    chigh[i] += 1
                else:
                    chigh[i] = 0
                if  ewma[i] < llimit:
                    clow[i] += 1
                else:
                    clow[i] = 0
              
                #only detect down trend for accuracy at the moment
                if clow[i] >= step_size and ewma[i] < llimit:
                    self._param['status'] = 'drifted'
                    if ewma[i] < llimit:
                        self._param['trend'] = 'down'
                    else:
                        self._param['trend'] = 'up'
                    self._param['type'] = 'label'
                    self._param['index'] = i
                    self._param['query_index'] = y[i][j][0]
                    return True, self._param['query_index']
        self._param['label']['counter'] += len(X)
        return False, None

    def _load_model(self, trial_id):
        trial = self._db.get_trial(trial_id)
        model = self._db.get_model(trial.model_id)

        # Load model based on trial
        clazz = load_model_class(model.model_file_bytes, model.model_class)
        model_inst = clazz()
        model_inst.init(trial.knobs)
        model_inst.load_parameters(trial.parameters)
        
        return model_inst

    def destroy(self):
        pass

    def dump_parameters(self, logger):
        return json.dumps(self._param)

    def load_parameters(self, param_str):
        self._param = json.loads(param_str)

    def _load_dataset_training(self, uri, task, model = None, logger = None):
        # Here, we use drift detection model's in-built dataset loader
        (X, y, yp) = load_dataset_training(uri, task, model)
        X = self._prepare_X(X)
        if self._param['status'] == 'uninitialized':
            self._param['ncol'] = len(X[0])
            self._param['nclass'] = len(np.unique(y))

        if (y is not None) and (yp is not None):
            y = self._prepare_y(y, yp)
        else:
            y = None
        return (X, y)

    def _prepare_X(self, images):
        return [np.array(image).flatten() for image in images]
    
    def _prepare_y(self, y, yp):
        result = [np.zeros(0) for i in range(self._param['nclass'])]
        for i in range(len(y)):
            if (y[i] == yp[i]):
                result[y[i]] = np.append(result[y[i]], 1.0)
            else:
                result[y[i]] = np.append(result[y[i]], 0.0)
        return np.array(result)
    
    def _load_prediction_from_feedbacks(self, nclass, trial_id, feedbacks):
        #returns a list of list
        #1st dimension is class number
        #2nd dimension is each feedback with that class number as true label
        #each element is a tuple (query_index, prediction) sorted by query_index
        self._db.connect()
        trial_mapping = self._db.get_trial_label_mapping(trial_id)
        inv_mapping = {v: int(k) for (k, v) in trial_mapping.items()}
        y = []
        for i in range(nclass):
            y.append([])
        for f in feedbacks:
            if self._db.get_prediction_by_index_and_trial(f[0], trial_id) == f[1]:
                y[inv_mapping[f[1]]].append((f[0], 1.0))
            else:
                y[inv_mapping[f[1]]].append((f[0], 0.0))
        for i in range(nclass):
            y[i].sort()
        self._db.disconnect()
        return y