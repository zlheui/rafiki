class BudgetType():
    MODEL_TRIAL_COUNT = 'MODEL_TRIAL_COUNT'

class InferenceJobStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'

class TrainJobStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    COMPLETED = 'COMPLETED'

class TrialStatus():
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    TERMINATED = 'TERMINATED'
    COMPLETED = 'COMPLETED'

class ServiceStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'

class ServiceType():
    TRAIN = 'TRAIN'
    PREDICT = 'PREDICT'
    INFERENCE = 'INFERENCE'

class UserType():
    SUPERADMIN = 'SUPERADMIN'
    ADMIN = 'ADMIN'
    MODEL_DEVELOPER = 'MODEL_DEVELOPER'
    APP_DEVELOPER = 'APP_DEVELOPER'
    USER = 'USER'

class AdvisorType():
    BTB_GP = 'BTB_GP'

class DatasetProtocol():
    HTTP = 'http'
    HTTPS = 'https'

class TaskType():
    IMAGE_CLASSIFICATION = 'IMAGE_CLASSIFICATION'