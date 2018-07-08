from gw000/keras:2.1.4-py3-tf-gpu

MAINTAINER Ollin Demian Langle Chimal <ollin.langle@ciencias.unam.mx>

ENV REFRESHED_AT 2018-07-05

COPY requirements.txt requirements.txt

RUN apt update && pip3 install -r requirements.txt

ENTRYPOINT /bin/bash
