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
COPY rafiki/container/requirements.txt container/requirements.txt
RUN pip install -r container/requirements.txt
COPY rafiki/data_repository/requirements.txt data_repository/requirements.txt
RUN pip install -r data_repository/requirements.txt

COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_data_repository.py start_data_repository.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/rafiki/

EXPOSE 8007

CMD ["python", "start_data_repository.py"]