FROM python:latest

# Create working directory and copy all necessary scripts
WORKDIR /app
COPY requirements.txt .
COPY preprocess_data.py .
COPY train.py .
COPY hpo.py .
COPY register_model.py .
COPY green_taxi_trip_predictor.py .

# Install all libraries
RUN apt-get -y update
RUN apt-get -y install sqlite3 libsqlite3-dev git

# Installl all dependencies
RUN pip install -r requirements.txt

# Copy needed data in parquet format and preprocess.
RUN mkdir data
COPY *.parquet ./data/

CMD ["python", "green_taxi_trip_predictor.py"]