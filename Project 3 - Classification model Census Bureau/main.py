'''
Main module to run the pipeline

Author: Sebastian Obando Morales
Date: June 19, 2022
'''

import pandas as pd
import joblib

from ml.classification_library import process_data,train_val_test,train_model,compute_model_metrics,inference

def pipeline():

    # Declare Variables
    categorical_variables = [' workclass',' education',' marital-status',' occupation',' relationship',' race',' native-country']
    binary_variables = [' sex',' salary']

    # Step 1: load the data
    data = pd.read_csv("./data/census.csv")
    # Step 2: processing
    df = process_data(data,binary_variables,categorical_variables)
    # Step 3: split data
    X_train, X_test, y_train, y_test = train_val_test(df)
    # Step 4: train the model
    model = train_model(X_train, y_train)
    ## Save the model
    joblib.dump(model, "./model/xgboost_model.joblib") 
    # Step 5: test the model
    ## Load the model
    model = joblib.load("./model/xgboost_model.joblib")
    preds = inference(model, X_test)
    # Step 6: evaluate the model
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print("precision:",precision)
    print("recall:",recall)
    print("fbeta:",fbeta)

if __name__ == "__main__":

    pipeline()
