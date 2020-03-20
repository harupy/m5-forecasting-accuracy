# Latest on 2020/03/20.
ARG BASE_TAG=2020.02
FROM continuumio/anaconda3:${BASE_TAG}

ARG WORKDIR=/m5-forecasting-accuracy
WORKDIR ${WORKDIR}

RUN apt-get update -y && \
    apt-get install -y unzip

# Copy requirements files.
COPY ./requirements.txt /tmp/requirements.txt
COPY ./requirements-dev.txt /tmp/requirements-dev.txt

# Install requirements.
RUN pip install -r /tmp/requirements.txt -r /tmp/requirements-dev.txt
