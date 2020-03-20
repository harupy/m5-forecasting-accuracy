# Latest on 2020/03/20.
ARG BASE_TAG=2020.02
FROM continuumio/anaconda3:${BASE_TAG}

ARG WORKDIR=/m5-forecasting-accuracy
WORKDIR ${WORKDIR}

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt
