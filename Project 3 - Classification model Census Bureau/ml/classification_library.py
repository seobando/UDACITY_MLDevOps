'''
Model functions

Author: Sebastian Obando Morales
Date: June 19, 2022
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score
import xgboost as xgb

def encode_binary_variables(df,binary_variable):

    le = LabelEncoder()
    df[binary_variable] = le.fit_transform(df[binary_variable])

    return df

def process_data(df,binary_variables,categorical_variables):

    ## Remove unrequired variables
    df = df.drop(' fnlgt',axis=1)
    ## Encode binary variables
    for binary_variable in binary_variables:
        df = encode_binary_variables(df,binary_variable)
    ## Encode catagorical variables
    df = pd.get_dummies(df,columns=categorical_variables)
    
    return df

def train_val_test(df):

    X_data = df.drop(' salary',axis = 1)    
    y_data = df[' salary']
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size= 0.3, random_state=42)
   
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    xg_cl = xgb.XGBClassifier(n_estimators=10,seed=42,use_label_encoder =False,eval_metric='logloss')
    
    return xg_cl.fit(X_train,y_train)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : xgboost
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return  preds