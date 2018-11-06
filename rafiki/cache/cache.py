import redis
import os
import json
import uuid

RUNNING_INFERENCE_WORKERS = 'INFERENCE_WORKERS'
QUERIES_QUEUE = 'QUERIES'
PREDICTIONS_QUEUE = 'PREDICTIONS'
RUNNING_DRIFT_DETECTION_WORKERS = 'DRIFT_DETECTION_WORKERS'

class Cache(object):
    def __init__(self,
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=os.environ.get('REDIS_PORT', 6379)):

        cache_connection_url = self._make_connection_url(
            host=host,
            port=port
        )

        self._connection_pool = redis.ConnectionPool.from_url(cache_connection_url)
        self._redis = redis.StrictRedis(connection_pool=self._connection_pool, decode_responses=True)

    def add_worker_of_inference_job(self, worker_id, inference_job_id):
        inference_workers_key = '{}_{}'.format(RUNNING_INFERENCE_WORKERS, inference_job_id)
        self._redis.sadd(inference_workers_key, worker_id)

    def delete_worker_of_inference_job(self, worker_id, inference_job_id):
        inference_workers_key = '{}_{}'.format(RUNNING_INFERENCE_WORKERS, inference_job_id)
        self._redis.srem(inference_workers_key, worker_id)

    def get_workers_of_inference_job(self, inference_job_id):
        inference_workers_key = '{}_{}'.format(RUNNING_INFERENCE_WORKERS, inference_job_id)
        worker_ids = self._redis.smembers(inference_workers_key)
        return [x.decode() for x in worker_ids]

    def add_drift_detection_worker(self, worker_id):
        drift_detection_workers_key = RUNNING_DRIFT_DETECTION_WORKERS
        self._redis.sadd(drift_detection_workers_key, worker_id)

    def delete_drift_detection_worker(self, worker_id):
        drift_detection_workers_key = RUNNING_DRIFT_DETECTION_WORKERS
        self._redis.srem(drift_detection_workers_key, worker_id)

    def get_drift_detection_workers(self):
        drift_detection_workers_key = RUNNING_DRIFT_DETECTION_WORKERS
        worker_ids = self._redis.smembers(drift_detection_workers_key)
        return [x.decoder() for x in worker_ids]

    def add_query_of_worker(self, worker_id, query):
        query_id = str(uuid.uuid4())
        query = json.dumps({
            'id': query_id,
            'query': query
        })

        worker_queries_key = '{}_{}'.format(QUERIES_QUEUE, worker_id)
        self._redis.rpush(worker_queries_key, query)
        return query_id

    def pop_queries_of_worker(self, worker_id, batch_size):
        worker_queries_key = '{}_{}'.format(QUERIES_QUEUE, worker_id)
        queries = self._redis.lrange(worker_queries_key, 0, batch_size - 1)
        self._redis.ltrim(worker_queries_key, len(queries), -1)
        queries = [json.loads(x) for x in queries]
        query_ids = [x['id'] for x in queries]
        queries = [x['query'] for x in queries]
        return (query_ids, queries)

    def add_prediction_of_worker(self, worker_id, query_id, prediction):
        prediction = json.dumps({
            'id': query_id,
            'prediction': prediction
        })

        worker_predictions_key = '{}_{}'.format(PREDICTIONS_QUEUE, worker_id)
        self._redis.rpush(worker_predictions_key, prediction)

    def pop_prediction_of_worker(self, worker_id, query_id):
        # Search through worker's list of predictions
        worker_predictions_key = '{}_{}'.format(PREDICTIONS_QUEUE, worker_id)
        predictions = self._redis.lrange(worker_predictions_key, 0, -1)

        for (i, prediction) in enumerate(predictions):
            prediction = json.loads(prediction)
            # If prediction for query found, remove prediction from list and return it
            if prediction['id'] == query_id:
                self._redis.ltrim(worker_predictions_key, i + 1, i)
                return prediction['prediction']

        # Return None if prediction is not found
        return None

    def _make_connection_url(self, host, port):
        return 'redis://{}:{}'.format(host, port)
