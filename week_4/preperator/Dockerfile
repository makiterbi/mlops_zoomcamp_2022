FROM python:3.9.7-slim

WORKDIR /app
COPY model.bin ./app/
COPY starter.ipynb ./app/

RUN python -m pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install scikit-learn==1.0.2 scikit-learn-intelex jupyter pandas pyarrow
