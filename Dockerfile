from gw000/keras:2.1.4-py3-tf-gpu

MAINTAINER Ollin Demian Langle Chimal <ollin.langle@ciencias.unam.mx>

ENV REFRESHED_AT 2018-07-05

RUN apt-get update && apt-get install -y curl apt-transport-https ca-certificates debian-keyring debian-archive-keyring

# RUN curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list

RUN apt-get update

RUN ACCEPT_EULA=Y apt-get install -y --allow-unauthenticated msodbcsql17 mssql-tools unixodbc-dev

RUN touch ~/.bashrc & echo 'export PATH="$PATH:/opt/mssql-tools/bin"' > ~/.bashrc & exec $SHELL

COPY requirements.txt requirements.txt

COPY data/.env /data/.env

RUN gpg --keyserver subkeys.pgp.net --recv-keys EB3E94ADBE1229CF

RUN gpg -a --export EB3E94ADBE1229CF | sudo apt-key add -
# RUN curl -sS https://packages.microsoft.com/debian/9/prod/dists/stretch/Release.gpg | apt-key add -

RUN apt update && pip3 install -r requirements.txt

ENTRYPOINT /bin/bash
