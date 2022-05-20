FROM python:latest

# Create working directory and copy all necessary scripts
WORKDIR /app
COPY nyc_taxi_predictor.py .
COPY requirements.txt .

# Installl all dependencies
RUN pip install -r requirements.txt

# Install wget and then download needed data for January and February 2021 in parquet format
RUN apt-get update & apt-get install wget
RUN wget https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-01.parquet & wget https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet
RUN mkdir data
RUN mv *.parquet data/

CMD ["python", "nyc_taxi_predictor.py"]