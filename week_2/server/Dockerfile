FROM python:latest

# Create working directory and copy all necessary scripts
WORKDIR /server
COPY requirements.txt .

# Install all libraries
RUN apt-get -y update
RUN apt-get -y install sqlite3 libsqlite3-dev git

# Installl all dependencies
RUN pip install -r requirements.txt

# Create Artifacts Repo
RUN mkdir mlruns

EXPOSE 5000