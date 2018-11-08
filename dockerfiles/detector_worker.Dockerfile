FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/rafiki/
WORKDIR /root/rafiki/

# Install python dependencies
COPY rafiki/utils/requirements.txt utils/requirements.txt
RUN pip install -r utils/requirements.txt
COPY rafiki/db/requirements.txt db/requirements.txt
RUN pip install -r db/requirements.txt
COPY rafiki/cache/requirements.txt cache/requirements.txt
RUN pip install -r cache/requirements.txt
COPY rafiki/drift_detection_method/requirements.txt drift_detection_method/requirements.txt
RUN pip install -r drift_detection_method/requirements.txt
COPY rafiki/client/requirements.txt client/requirements.txt
RUN pip install -r client/requirements.txt
COPY rafiki/worker/requirements.txt worker/requirements.txt
RUN pip install -r worker/requirements.txt


COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_detector_worker.py start_detector_worker.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/rafiki/

ENTRYPOINT [ "python", "start_detector_worker.py" ]