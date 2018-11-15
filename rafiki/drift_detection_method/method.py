import os
import json
import abc

from rafiki.db import Database
from rafiki.cache import Cache
from rafiki.constants import ServiceType

class BaseMethod(abc.ABC):
    '''
    Rafiki's base concept drift detection method class that Rafiki detection methods should extend. 
    Rafiki detection methods should implement all abstract methods.
    '''

    def init(self, cache=Cache(), db=Database(isolation_level='REPEATABLE_READ')):
        self._cache = cache
        self._db = db

    @abc.abstractmethod
    def update_on_queries(self, train_job_id, queries, query_index=None):
        raise(NotImplementedError)

    @abc.abstractmethod
    def update_on_feedbacks(self, train_job_id, feedbacks):
        raise(NotImplementedError)

    def upload_queries(self, train_job_id, queries, output, logger):
        logger.info('upload data')
        self._db.connect()
        train_job = self._db.get_train_job(train_job_id)
        query_stats = self._db.get_query_stats_by_train_job_id(train_job_id)
        
        logger.info('update query_stats')
        if query_stats is None:
            query_stats = self._db.create_query_stats(train_job_id)
            self._db.commit()
        
        query_index = query_stats.next_query_index
        self._db.update_query_stats(query_stats, query_index+len(queries))
        self._db.commit()
        self._db.disconnect()

        logger.info('upload use cache')
        # use cache to send the queries to data repository, only one worker is available
        # scalability is ensured by replicas of the service
        running_data_repository_worker_ids = self._cache.get_data_repository_workers(service_type=ServiceType.REPOSITORY_QUERY)
        logger.info('worker_id: ' + ','.join(running_data_repository_worker_ids))
        if len(running_data_repository_worker_ids) > 0:
            query_index = self._cache.add_queries_of_data_repository_worker(running_data_repository_worker_ids[0], train_job_id, query_index, queries)
        
        logger.info('finish uploading')

        output.put((train_job_id, query_index))
        return (train_job_id, query_index)
