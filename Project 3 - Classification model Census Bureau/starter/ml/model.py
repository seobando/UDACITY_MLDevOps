'''
Model functions

Author: Sebastian Obando Morales
Date: June 19, 2022
'''

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold
import xgboost as xgb
from joblib import dump,load

from ml.constants import path_model,path_slices

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
    # Train the model
    xg_cl = xgb.XGBClassifier(n_estimators=10,seed=42,use_label_encoder =False,eval_metric='logloss')
    model = xg_cl.fit(X_train,y_train)    

    # Save the model
    dump(model, path_model) 

    return model


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


def inference(X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Load Model
    model = load(path_model)
    # Apply Prediction
    preds = model.predict(X)

    return  preds

def measure_model_performance(df, X, y, categorical_variable):
    """ Measure the model performance using slices of the data
  
    Inputs
    ------
    df:

    X : np.array
        Data used for prediction.

    y : np.array
  
    categorical_variable:

    """
    slice_values = []

    for categorical_value in df[categorical_variable].unique():
        indexes = (df[categorical_variable] == categorical_value)
        indexes_preds = inference(X[indexes])
        precision, recall, fbeta = compute_model_metrics(y[indexes], indexes_preds)

        line = "[%s->%s] Precision: %s Recall: %s FBeta: %s" % (categorical_variable,categorical_value, precision, recall, fbeta)
        slice_values.append(line)

    with open(path_slices, 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')        
        
    