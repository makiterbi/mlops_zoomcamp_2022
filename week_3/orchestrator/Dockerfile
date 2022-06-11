FROM python:latest

# Create working directory and copy all necessary scripts
WORKDIR /app
COPY requirements.txt .

# Install all dependencies
RUN pip install -r requirements.txt

# Install all libraries
RUN apt-get -y update
RUN apt-get -y install sqlite3 libsqlite3-dev

# Configure Prefect 
RUN prefect config set PREFECT_ORION_UI_API_URL="http://orchestrator:4200/api"