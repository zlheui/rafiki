# Global
APP_SECRET = 'rafiki'
SUPERADMIN_EMAIL = 'superadmin@rafiki'
SUPERADMIN_PASSWORD = 'rafiki'

# Admin
MIN_SERVICE_PORT = 30000
MAX_SERVICE_PORT = 32767
TRAIN_WORKER_REPLICAS_PER_MODEL = 2
INFERENCE_WORKER_REPLICAS_PER_TRIAL = 2
INFERENCE_MAX_BEST_TRIALS = 2

# Predictor
PREDICTOR_PREDICT_SLEEP = 0.25

# Inference worker
INFERENCE_WORKER_SLEEP = 0.25
INFERENCE_WORKER_PREDICT_BATCH_SIZE = 32