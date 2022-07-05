from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 
labels = config['labels']
features = config['features']

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    ## Declare variables
    testdata = pd.read_csv(test_data_path + '/testdata.csv')
    X = testdata.loc[:,features].values.reshape(-1, len(features))
    y = testdata[labels].values.reshape(-1, 1).ravel()
    ## Get model
    with open(model_path + "/trainedmodel.pkl", 'rb') as file:
        model = pickle.load(file)
    ## Get predictions
    predicted = model.predict(X)
    ## Get f1score
    f1score = metrics.f1_score(predicted,y)
    ## Save score
    with open(model_path + "/latestscore.txt", 'w') as f:
        f.write(str(f1score) + "\n")

    return f1score

if __name__ == "__main__":
    score_model()   