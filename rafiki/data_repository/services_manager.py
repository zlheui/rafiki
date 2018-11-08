import os
import logging
import traceback

from rafiki.db import Database
from rafiki.constants import ServiceStatus, ServiceType
from rafiki.config import MIN_SERVICE_PORT, MAX_SERVICE_PORT, DATA_REPOSITORY_WORKER_REPLICAS

from rafiki.container import DockerSwarmContainerManager 

logger = logging.getLogger(__name__)

class ServicesManager(object):
    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._db = db
        self._container_manager = container_manager
        self._service_id = None

    def create_data_repository_service(self, service_type=ServiceType.DATA_REPOSITORY):
        replicas = self._compute_data_repository_worker_replicas()

        environment_vars = {
            'POSTGRES_HOST': os.environ['POSTGRES_HOST'],
            'POSTGRES_PORT': os.environ['POSTGRES_PORT'],
            'POSTGRES_USER': os.environ['POSTGRES_USER'],
            'POSTGRES_DB': os.environ['POSTGRES_DB'],
            'POSTGRES_PASSWORD': os.environ['POSTGRES_PASSWORD'],
            'LOGS_FOLDER_PATH': os.environ['LOGS_FOLDER_PATH'],
            'REDIS_HOST': os.environ['REDIS_HOST'],
            'REDIS_PORT': os.environ['REDIS_PORT'],
            'ADMIN_HOST': os.environ['ADMIN_HOST'],
            'ADMIN_PORT': os.environ['ADMIN_PORT'],
            'ADVISOR_HOST': os.environ['ADVISOR_HOST'],
            'ADVISOR_PORT': os.environ['ADVISOR_PORT']
        }

        service = self._create_service(
            service_type=service_type,
            docker_image=os.environ['RAFIKI_IMAGE_DATA_REPOSITORY_WORKER']+':'+os.environ['RAFIKI_VERSION'],
            replicas=replicas,
            environment_vars=environment_vars
        )

        self._service_id = service.id

        return service

    def stop_data_repository_service(self):
        if self._service_id is not None:
            service = self._db.get_service(self._service_id)
            self._stop_service(service)

        return self._service_id

    def _stop_service(self, service):
        if service.container_service_id is not None:
            self._container_manager.destroy_service(service.container_service_id)

        self._db.mark_service_as_stopped(service)
        self._db.commit()

    def _create_service(self, service_type, docker_image,
                        replicas, environment_vars={}, args=[], 
                        container_port=None):
        
        # Create service in DB
        container_manager_type = type(self._container_manager).__name__
        service = self._db.create_service(
            container_manager_type=container_manager_type,
            service_type=service_type,
            docker_image=docker_image
        )
        self._db.commit()

        # Pass service details as environment variables 
        environment_vars = {
            **environment_vars,
            'RAFIKI_SERVICE_ID': service.id,
            'RAFIKI_SERVICE_TYPE': service_type
        }

        # Mount logs folder onto workers too
        logs_folder_path = os.environ['LOGS_FOLDER_PATH']
        mounts = {
            logs_folder_path: logs_folder_path
        }

        # Expose container port if it exists
        publish_port = None
        ext_hostname = None
        ext_port = None
        if container_port is not None:
            ext_hostname = os.environ['RAFIKI_IP_ADDRESS']
            ext_port = self._get_available_ext_port()
            publish_port = (ext_port, container_port)

        try:
            container_service_name = 'rafiki_service_{}'.format(service.id)
            container_service = self._container_manager.create_service(
                service_name=container_service_name,
                docker_image=docker_image, 
                replicas=replicas, 
                args=args,
                environment_vars=environment_vars,
                mounts=mounts,
                publish_port=publish_port
            )
            
            container_service_id = container_service['id']
            hostname = container_service['hostname']
            port = container_service.get('port', None)

            self._db.mark_service_as_running(
                service,
                container_service_name=container_service_name,
                container_service_id=container_service_id,
                replicas=replicas,
                hostname=hostname,
                port=port,
                ext_hostname=ext_hostname,
                ext_port=ext_port
            )
            self._db.commit()

        except Exception:
            logger.error('Error while creating service with ID {}'.format(service.id))
            logger.error(traceback.format_exc())
            self._db.mark_service_as_errored(service)
            self._db.commit()

        return service


    # Compute next available external port
    def _get_available_ext_port(self):
        services = self._db.get_services(status=ServiceStatus.RUNNING)
        used_ports = [int(x.ext_port) for x in services if x.ext_port is not None]
        port = MIN_SERVICE_PORT
        while port <= MAX_SERVICE_PORT:
            if port not in used_ports:
                return port

            port += 1

        return port

    def _compute_data_repository_worker_replicas():
        return DATA_REPOSITORY_WORKER_REPLICAS

