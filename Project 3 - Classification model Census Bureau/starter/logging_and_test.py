'''
Test module

Author: Sebastian Obando Morales
Date: July 07, 2022
'''

import logging
import os
import pytest
from sklearn.model_selection import train_test_split
from starter.ml.data import load_data, process_data
from starter.ml.model import train_model, inference, compute_model_metrics
from starter.ml.constants import path_name,categorical_variables,path_model_logs,path_model_folder

logging.basicConfig(
    filename=path_model_logs,
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture
def data():
    """
    Get required data
    """

    df = load_data(path_name)

    train, test = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=categorical_variables, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=categorical_variables, label="salary", training=False, encoder=encoder, lb=lb
    )

    return X_train, y_train, X_test, y_test

def test_train_model(data):
    '''
    test train_models
    '''

    X_train, y_train, X_test, y_test = data

    expected_model = ['model.joblib']
    
    try:
        train_model(X_train, y_train)
        existing_file = os.listdir(path_model_folder)
        assert set(expected_model).issubset(existing_file)
        logging.info("The expected models are there :SUCCESS")
    except AssertionError as err:
        logging.error("The expected models are not there :ERROR")
        raise err

def test_inference(data):
    """
    test inference
    """

    X_train, y_train, X_test, y_test = data

    try:
        preds = inference(X_test)
        assert len(preds.tolist()) > 0
        logging.info("The prediction is done :SUCCESS")
    except AssertionError as err:
        logging.error("The prediction is not done :ERROR")
        raise err       

## Last required modifications ##
@pytest.fixture
def label_values():
    dummy_label_values = [0,1,0,0,0,1,0,1,0,1]
    return dummy_label_values

@pytest.fixture
def pred_values():
    dummy_pred_values = [0,0,0,1,0,1,1,1,0,1]
    return dummy_pred_values    

def test_compute_model_metrics(label_values,pred_values):
    """
    test compute model metrics
    """
    y = label_values
    preds = pred_values

    try:
        precision, recall, fbeta = compute_model_metrics(y, preds)
        assert precision != 0
        assert recall != 0
        assert fbeta != 0
        logging.info("The metrics are ok :SUCCESS")
    except AssertionError as err:
        logging.error("The metrics must be checked :ERROR")
        raise err  

