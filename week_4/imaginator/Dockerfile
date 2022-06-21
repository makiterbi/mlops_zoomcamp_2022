FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

COPY starter.py .

RUN python -m pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install scikit-learn==1.0.2 scikit-learn-intelex jupyter pandas pyarrow
