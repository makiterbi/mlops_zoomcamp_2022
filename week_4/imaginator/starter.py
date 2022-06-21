import json
import os
import pickle
import pandas as pd
import argparse


def load_model():
    with open('model.bin', 'rb') as f_in:
        return pickle.load(f_in)


def read_data(filename, categorical):

    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def run(year, month):

    categorical = ['PUlocationID', 'DOlocationID']
    data_path = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet"

    print(
        f"Data at {data_path}  will be downloaded for {year} / {month}")

    df = read_data(data_path, categorical)

    dv, lr = load_model()
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    mean_pred = sum(list(y_pred)) / len(list(y_pred))
    print(
        f"Mean of the predictions for {year} / {month} = {round(mean_pred, 2)}")

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = df[['ride_id']]

    df_result['predictions'] = y_pred

    df_result.to_parquet(
        'df_results.parquet',
        engine='pyarrow',
        compression=None,
        index=False
    )

    file_size = os.path.getsize('df_results.parquet') / (1024*1024)
    print("Size of the result file =", round(file_size, 0))

    f = open('Pipfile.lock')
    data = json.load(f)
    first_hash = data['default']['scikit-learn']['hashes'][0]
    print("First hash of scikit-learn =", first_hash)

    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        default=2021,
        help="The year of the collected data"
    )
    parser.add_argument(
        "--month",
        default=2,
        help="The month of the collected data"
    )
    args = parser.parse_args()

    run(int(args.year), int(args.month))
