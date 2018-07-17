from gw000/keras:2.1.4-py3-tf-gpu

MAINTAINER Ollin Demian Langle Chimal <ollin.langle@ciencias.unam.mx>

ENV REFRESHED_AT 2018-07-05

RUN curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

RUN apt-get update & ACCEPT_EULA=Y apt-get install msodbcsql17 & ACCEPT_EULA=Y apt-get install mssql-tools

RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc & source ~/.bashrc

RUN sudo apt-get install unixodbc-dev

COPY requirements.txt requirements.txt

COPY data/.env /data/.env

RUN apt update && pip3 install -r requirements.txt

ENTRYPOINT /bin/bash
