from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, ForeignKey, Integer, Binary, DateTime, Boolean, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSON
import uuid
import datetime

from rafiki.constants import InferenceJobStatus, ServiceStatus, TrainJobStatus, \
    TrialStatus

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

def generate_datetime():
    return datetime.datetime.utcnow()

class InferenceJob(Base):
    __tablename__ = 'inference_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    status = Column(String, nullable=False, default=InferenceJobStatus.STARTED)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    predictor_service_id = Column(String, ForeignKey('service.id'))
    datetime_stopped = Column(DateTime, default=None)

class InferenceJobWorker(Base):
    __tablename__ = 'inference_job_worker'

    service_id = Column(String, ForeignKey('service.id'), primary_key=True)
    inference_job_id = Column(String, ForeignKey('inference_job.id'))
    trial_id = Column(String, ForeignKey('trial.id'), nullable=False)

class Model(Base):
    __tablename__ = 'model'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_created = Column(DateTime, nullable=False, default=generate_datetime)
    name = Column(String, unique=True, nullable=False)
    task = Column(String, nullable=False)
    model_file_bytes = Column(Binary, nullable=False)
    model_class = Column(String, nullable=False)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    docker_image = Column(String, nullable=False)


class Service(Base):
    __tablename__ = 'service'

    id = Column(String, primary_key=True, default=generate_uuid)
    service_type = Column(String, nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    datetime_stopped = Column(DateTime, default=None)
    status = Column(String, nullable=False, default=ServiceStatus.STARTED)
    docker_image = Column(String, nullable=False)
    container_manager_type = Column(String, nullable=False)
    replicas = Column(Integer, default=0)
    ext_hostname = Column(String)
    ext_port = Column(Integer)
    hostname = Column(String)
    port = Column(Integer)
    container_service_name = Column(String)
    container_service_id = Column(String)


class TrainJob(Base):
    __tablename__ = 'train_job'

    id = Column(String, primary_key=True, default=generate_uuid)
    app = Column(String, nullable=False)
    app_version = Column(Integer, nullable=False)
    task = Column(String, nullable=False)
    train_dataset_uri = Column(String, nullable=False)
    test_dataset_uri = Column(String, nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    budget_type = Column(String, nullable=False)
    budget_amount = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default=TrainJobStatus.STARTED)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    datetime_completed = Column(DateTime, default=None)
    subscribe_to_drift_detection_service = Column(Boolean, default=False)

class TrainJobWorker(Base):
    __tablename__ = 'train_job_worker'

    service_id = Column(String, ForeignKey('service.id'), primary_key=True)
    train_job_id = Column(String, ForeignKey('train_job.id'))
    model_id = Column(String, ForeignKey('model.id'), nullable=False)


class Trial(Base):
    __tablename__ = 'trial'

    id = Column(String, primary_key=True, default=generate_uuid)
    knobs = Column(JSON, nullable=False)
    datetime_started = Column(DateTime, nullable=False, default=generate_datetime)
    train_job_id = Column(String, ForeignKey('train_job.id'), nullable=False)
    model_id = Column(String, ForeignKey('model.id'), nullable=False)
    status = Column(String, nullable=False, default=TrialStatus.RUNNING)
    score = Column(Float, default=0)
    parameters = Column(JSON, default=None)
    predict_label_mapping = Column(JSON, default=None)
    datetime_stopped = Column(DateTime, default=None)
    

class User(Base):
    __tablename__ = 'user'

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(Binary, nullable=False)
    user_type = Column(String, nullable=False)


class QueryStats(Base):
    __tablename__ = 'query_stats'

    id = Column(String, primary_key=True, default=generate_uuid)
    train_job_id = Column(String, ForeignKey('train_job.id'), unique=True)
    next_query_index = Column(Integer, nullable=False)

class Prediction(Base):
    __tablename__ = 'prediction'

    id = Column(String, primary_key=True, default=generate_uuid)
    query_index = Column(Integer, nullable=False)
    trial_id = Column(String, ForeignKey('trial.id'), nullable=False)
    predict = Column(String, nullable=False)
    UniqueConstraint('query_index', 'trial_id', name='uix_query_trial')

class Feedback(Base):
    __tablename__ = 'feedback'

    id = Column(String, primary_key=True, default=generate_uuid)
    query_index = Column(Integer, nullable=False)
    label = Column(String, nullable=False) 

class Detector(Base):
    __tablename__ = 'detector'

    id = Column(String, primary_key=True, default=generate_uuid)
    datetime_created = Column(DateTime, nullable=False, default=generate_datetime)
    user_id = Column(String, ForeignKey('user.id'), nullable=False)
    name = Column(String, unique=True, nullable=False)
    detector_file_bytes = Column(Binary, nullable=False)

class DriftDetectionSub(Base):
    __tablename__ = 'drift_detection_subscription'

    id = Column(String, primary_key=True, default=generate_uuid)
    train_job_id = Column(String, ForeignKey('train_job.id'), nullable=False)
    detector_name = Column(String, ForeignKey('detector.name'), nullable=False)
    UniqueConstraint('train_job_id', 'detector_name', name='uix_train_job_detector')