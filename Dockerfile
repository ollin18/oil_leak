from gw000/keras:2.1.4-py3-tf-gpu

MAINTAINER Ollin Demian Langle Chimal <ollin.langle@ciencias.unam.mx>

ENV REFRESHED_AT 2018-07-05

RUN apt-get update && apt-get install -y curl apt-transport-https ca-certificates debian-keyring debian-archive-keyring

RUN curl -sS https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

RUN curl -sS https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssqlrelease.list

RUN curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list

RUN apt-get update

RUN ACCEPT_EULA=Y apt-get install -y --allow-unauthenticated msodbcsql17 msodbcsql mssql-tools unixodbc-dev unixodbc odbcinst1debian2

RUN touch ~/.bashrc & echo 'export PATH="$PATH:/opt/mssql-tools/bin"' > ~/.bashrc & exec $SHELL

COPY requirements.txt requirements.txt

COPY data/.env /data/.env

RUN apt update && pip3 install -r requirements.txt

ENTRYPOINT /bin/bash
