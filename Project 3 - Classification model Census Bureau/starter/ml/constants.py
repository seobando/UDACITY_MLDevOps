'''
A module to hold the required constants

Author: Sebastian Obando Morales
Date: June 20, 2022
'''

path_name = "../data/census.csv"

path_model = "../model/xgboost_model.joblib"

path_encoder = "../model/encoder.joblib"

path_lb = "../model/lb.joblib"

path_slices = "../logs/slice_output.txt"

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