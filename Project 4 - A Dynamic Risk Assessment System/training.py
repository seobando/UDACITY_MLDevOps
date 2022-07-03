from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():

    with open('deployedmodelname.txt', 'r') as f:
        deployedname = f.read()
    print(deployedname)    

    with open('demodatalocation.txt', 'r') as f:
       datalocation = f.read()
    print(datalocation)   

    trainingdata=pd.read_csv(os.getcwd()+datalocation)
    X=trainingdata.loc[:,['bed','bath']].values.reshape(-1, 2)
    y=trainingdata['highprice'].values.reshape(-1, 1).ravel()

    #use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='warn', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model = logit.fit(X, y)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open('./production/'+deployedname, 'wb'))
