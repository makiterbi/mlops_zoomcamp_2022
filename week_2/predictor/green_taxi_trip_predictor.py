import os
import os.path
import subprocess
import argparse

import mlflow

# Q1. Install MLflow


def check_mlflow():
    print(f"Installed MLFlow Version is {mlflow.__version__}")

# Q2. Download and preprocess the data


def check_files():
    os.system(
        "python preprocess_data.py --raw_data_path ./data --dest_path ./output")
    size = len([name for name in os.listdir('./output')])
    print(f"Number of saved files to output directory is {size}")

# Q3. Train a model with autolog


def train():
    os.system("python train.py")

# Q4. Launch the tracking server locally


def launch():
    print('In addition to `backend-store-uri`, we need to pass `default-artifact-root` to properly configure the server!')


# Q5. Tune the hyperparameters of the model
def tune():
    os.system("python hpo.py")

# Q6. Promote the best model to the model registry


def promote():
    os.system("python register_model.py")

# Main flow


def run():
    check_mlflow()
    check_files()
    train()
    launch()
    tune()
    promote()
    print("All completed!")


# Regular entry point for python script
if __name__ == '__main__':
    #arg_parser = argparse.ArgumentParser()
    #arg_parser.add_argument('--experiment', action='store', type=str, required=True)
    #args = arg_parser.parse_args()
    run()
