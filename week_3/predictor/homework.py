import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task

from datetime import datetime
from dateutil.relativedelta import relativedelta

import pickle

import os
import os.path


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return


@task
def get_paths(date):

    if date is None:
        date = datetime.now()
    else:
        try:
            date = datetime. strptime(date, '%Y-%m-%d')
        except:
            print('Malformed date please use format with YYYY-MM-DD!')
    training_date = date + relativedelta(months=-2)
    val_date = date + relativedelta(months=-1)
    training_date_string = str(training_date.year) + \
        '-' + str(training_date.month).zfill(2)
    val_date_string = str(val_date.year) + '-' + str(val_date.month).zfill(2)
    return f"./data/fhv_tripdata_{training_date_string}.parquet", f"./data/fhv_tripdata_{val_date_string}.parquet"


@flow
def main(date=None):
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    with open(f'model-{date}.pkl', 'wb') as model_file:
        pickle.dump(lr, model_file)
    with open(f'dv-{date}.pkl', 'wb') as dv_file:
        pickle.dump(dv, dv_file)

    file_exists = os.path.exists(f'dv-{date}.pkl')
    if file_exists:
        file_stats = os.stat(f'dv-{date}.pkl')
        print(
            f'the file size of the DictVectorizer in bytes is {file_stats.st_size}')

    run_model(df_val_processed, categorical, dv, lr)


main(date='2021-08-15')
