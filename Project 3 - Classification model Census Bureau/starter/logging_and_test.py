'''
Test module

Author: Sebastian Obando Morales
Date: June 20, 2022
'''

import logging
import os
from sklearn.model_selection import train_test_split
from ml.data import load_data, process_data
from ml.model import train_model, inference, compute_model_metrics
from ml.constants import path_name,categorical_variables

logging.basicConfig(
    filename='../logs/model.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_train_model(X_train, y_train):
    '''
    test train_models
    '''
    expected_model = ['xgboost_model.joblib']
    
    try:
        train_model(X_train, y_train)
        existing_file = os.listdir("../model/")
        assert set(expected_model).issubset(existing_file)
        logging.info("The expected models are there :SUCCESS")
    except AssertionError as err:
        logging.error("The expected models are not there :ERROR")
        raise err

def test_inference(X):
    """
    test inference
    """
    try:
        preds = inference(X)
        assert len(preds.tolist()) > 0
        logging.info("The prediction is done :SUCCESS")
    except AssertionError as err:
        logging.error("The prediction is not done :ERROR")
        raise err       

    return preds

def test_compute_model_metrics(y, preds):
    """
    test compute model metrics
    """
    try:
        precision, recall, fbeta = compute_model_metrics(y, preds)
        assert precision != 0
        assert recall != 0
        assert fbeta != 0
        logging.info("The metrics are ok :SUCCESS")
    except AssertionError as err:
        logging.error("The metrics must be checked :ERROR")
        raise err  

if __name__ == "__main__":

    print("Phase 1: Load the Data\n")
    print("\n-------------------------------------------------------")
    data = load_data(path_name)
    print("Phase 2: Split the Data\n")
    print("\n-------------------------------------------------------")
    train, test = train_test_split(data, test_size=0.20)
    print("Phase 3: Process the training set\n")
    print("\n-------------------------------------------------------")
    # Proces the train data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=categorical_variables, label="salary", training=True
    )
    print("Phase 4: Process the testing set\n")
    print("\n-------------------------------------------------------")
    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        train, categorical_features=categorical_variables, label="salary", training=False, encoder=encoder, lb=lb
    )
    print("Phase 5: Train the model\n")
    print("\n-------------------------------------------------------")
    test_train_model(X_train, y_train)
    print("Phase 6: Classify\n")
    print("\n-------------------------------------------------------")
    preds = test_inference(X_test)
    print("Phase 7: Test the model\n")
    print("\n-------------------------------------------------------")
    test_compute_model_metrics(y_test, preds)


