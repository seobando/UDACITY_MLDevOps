'''
A module to hold the testing functions

Author: Sebastian Obando Morales
Date: May 15, 2022
'''


import os
import logging
import churn_library as cl
import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = cl.import_data(constants.path_name)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return data_frame


def test_eda(data_frame):
    '''
    test perform eda function
    '''
    eda_expeted_images = [
        "heatmap.png",
        "marital_status_distribution.png",
        "churn_distribution.png",
        "customer_age_distribution.png",
        "total_transactions_distribution.png"
    ]

    try:
        cl.perform_eda(data_frame, constants.cat_columns, constants.quant_columns)
        existing_files = os.listdir("./images/eda/")
        assert set(eda_expeted_images).issubset(existing_files)
        logging.info("The eda expected images are there :SUCCESS")
    except AssertionError as err:
        logging.error("The eda expected images are not there :ERROR")
        raise err


def test_encoder_helper(data_frame):
    '''
    test encoder helper
    '''
    expected_encode_columns = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ]

    try:
        encoder_helper = cl.encoder_helper(data_frame, constants.cat_columns)
        required_columns = encoder_helper.columns
        assert set(expected_encode_columns).issubset(required_columns)
        logging.info("The expected encode columns are there :SUCCESS")
    except AssertionError as err:
        logging.error("The expected encode columns are not there :ERROR")
        raise err

    return encoder_helper


def test_perform_feature_engineering(encoder_helper):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            encoder_helper, constants.keep_cols)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("The split works :SUCCESS")
    except AssertionError as err:
        logging.error("The split didn't work :ERROR")
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    expected_models = [
        'rfc_model.pkl',
        'logistic_model.pkl'
    ]
    try:
        cl.train_models(x_train, x_test, y_train, y_test)
        existing_files = os.listdir("./models/")
        assert set(expected_models).issubset(existing_files)
        logging.info("The expected models are there :SUCCESS")
    except AssertionError as err:
        logging.error("The expected models are not there :ERROR")
        raise err


if __name__ == "__main__":

    DATA_FRAME = test_import()
    test_eda(DATA_FRAME)
    ENCODER_HELPER = test_encoder_helper(DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        ENCODER_HELPER)
    test_train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
