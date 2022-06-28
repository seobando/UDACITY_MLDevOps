'''
A module to hold the required constants

Author: Sebastian Obando Morales
Date: June 20, 2022
'''
import os

is_development = 1

if is_development:

    ROOT_DIR = '../'

    file_dir = 'data/census.csv'

    path_name = f'{ROOT_DIR}/{file_dir}'

    model_dir = "model/model.joblib"

    path_model = f'{ROOT_DIR}/{model_dir}'

    encoder_dir = "model/encoder.joblib"

    path_encoder = f'{ROOT_DIR}/{encoder_dir}'

    lb_dir = "model/lb.joblib"

    path_lb = f'{ROOT_DIR}/{lb_dir}'

    slices_dir = "logs/slice_output.txt"

    path_slices = f'{ROOT_DIR}/{slices_dir}'

    model_logs_dir = "logs/model.log"

    path_model_logs = f'{ROOT_DIR}/{model_logs_dir}'

    model_folder_dir = "model/"

    path_model_folder = f'{ROOT_DIR}/{model_folder_dir }'

else:

    path_model = 'model/model.joblib'
    path_encoder = 'model/encoder.joblib'
    path_lb = 'model/lb.joblib'
    path_slices = 'logs/slice_output.txt'

categorical_variables = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
    ]

