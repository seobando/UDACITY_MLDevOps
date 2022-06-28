'''
A module to hold the required constants

Author: Sebastian Obando Morales
Date: June 20, 2022
'''
import os

is_development = 0

if is_development:

    ROOT_DIR = os.path.abspath(os.curdir)

    file_dir = 'Project 3 - Classification model Census Bureau/data/census.csv'

    path_name = f'{ROOT_DIR}/{file_dir}'

    model_dir = "Project 3 - Classification model Census Bureau/model/model.joblib"

    path_model = f'{ROOT_DIR}/{model_dir}'

    encoder_dir = "Project 3 - Classification model Census Bureau/model/encoder.joblib"

    path_encoder = f'{ROOT_DIR}/{encoder_dir}'

    lb_dir = "Project 3 - Classification model Census Bureau/model/lb.joblib"

    path_lb = f'{ROOT_DIR}/{lb_dir}'

    slices_dir = "Project 3 - Classification model Census Bureau/logs/slice_output.txt"

    path_slices = f'{ROOT_DIR}/{slices_dir}'

    model_logs_dir = "Project 3 - Classification model Census Bureau/logs/model.log"

    path_model_logs = f'{ROOT_DIR}/{model_logs_dir}'

    model_folder_dir = "Project 3 - Classification model Census Bureau/model/"

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

