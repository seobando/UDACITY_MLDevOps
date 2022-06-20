'''
A module to hold the required constants

Author: Sebastian Obando Morales
Date: June 20, 2022
'''

path_name = "../data/census.csv"

path_model = "../model/xgboost_model.joblib"

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