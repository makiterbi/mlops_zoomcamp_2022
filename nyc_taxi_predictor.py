#!/usr/bin/env python
# coding: utf-8

# Imports
import glob
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Method for loading data


def load_data():
    files = glob.glob("./data/*.parquet")
    chunks = [pd.read_parquet(f) for f in files]
    return pd.concat(chunks, ignore_index=True)

# Solution for Q1. Reporting number of records in January data.


def check_number_of_records(data: pd.DataFrame):
    size = data[data.pickup_datetime.dt.month == 1].shape[0]
    print(f"January data has {size} records!")

# Solution for Q2. Reporting mean of duration in January data.


def average_trip_duration(data, month):
    mean = data[data.pickup_datetime.dt.month == month].duration.mean()
    print(f"The average trip duration in month {month} is {mean}")

# Solution for Data Preparation. Reporting number of records dropped.


def filter_data_by_duration(data, month, lower_limit, upper_limit):
    former_size = data[data.pickup_datetime.dt.month == month].shape[0]
    data = data[(data.duration >= lower_limit) &
                (data.duration <= upper_limit)]
    number_of_removed_items = former_size - \
        data[(data.pickup_datetime.dt.month == month)].shape[0]
    print(f" {number_of_removed_items} records are eliminated!")
    return data

# Solution for Q3. Reporting the fractions of missing values for the pickup location ID.


def fill_missing_values(data, month, value):
    data.PUlocationID.fillna(value, inplace=True)
    data.DOlocationID.fillna(value, inplace=True)
    fraction = data[(data.pickup_datetime.dt.month == month) & (
        data.PUlocationID == -1.0)].shape[0] / data[data.pickup_datetime.dt.month == month].shape[0]
    print(
        f"The fractions of missing values for the pickup location ID is {fraction}")
    return data

# Solution for Q4. Reporting the dimensionality of encoded matrix.


def encode(data, month, verbose=False):
    dictionaries = data[data.pickup_datetime.dt.month == month][[
        "PUlocationID", "DOlocationID"]].to_dict(orient='records')
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(dictionaries)
    if verbose:
        print(f"The second dimensionality of the matrix is {X.shape[1]} .")
    y = data[data.pickup_datetime.dt.month == month].duration.values
    return X, y

# Solution for Q5 and Q6. Reporting the RMSE.


def fit_and_predict(model, X, y, is_training=False):
    if is_training:
        model.fit(X, y)
    y_pred = model.predict(X)
    rsme = mean_squared_error(y, y_pred, squared=False)
    print(f"Root mean squared error is {rsme}")

# Main flow of the script


def run():
    data = load_data()
    data.pickup_datetime = pd.to_datetime(data.pickup_datetime)
    data.dropOff_datetime = pd.to_datetime(data.dropOff_datetime)
    check_number_of_records(data)
    data["duration"] = (data.dropOff_datetime -
                        data.pickup_datetime).dt.total_seconds() / 60.0
    average_trip_duration(data, 1)
    data = filter_data_by_duration(data, 1, 1.0, 60.0)
    data = fill_missing_values(data, 1, -1.0)
    X_train, y_train = encode(data, 1, True)
    linear_regression_model = LinearRegression()
    fit_and_predict(linear_regression_model, X_train, y_train, True)
    X_val, y_val = encode(data, 2)
    fit_and_predict(linear_regression_model, X_val, y_val)


# Regular entry point for python script
if __name__ == '__main__':
    run()
