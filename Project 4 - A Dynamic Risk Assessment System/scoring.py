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

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    with open('samplemodel.pkl', 'rb') as file:
        model = pickle.load(file)

    testdata=pd.read_csv('testdata.csv')
    X = testdata[['col1','col2']].values.reshape(-1,2)
    y = testdata['col3'].values.reshape(-1,1)

    predicted = model.predict(X)

    f1score = metrics.f1_score(predicted,y)

    return f1score   