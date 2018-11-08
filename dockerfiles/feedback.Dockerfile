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
COPY rafiki/feedback/requirements.txt feedback/requirements.txt
RUN pip install -r feedback/requirements.txt

COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_feedback.py start_feedback.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/rafiki/

EXPOSE 8006

CMD ["python", "start_feedback.py"]
