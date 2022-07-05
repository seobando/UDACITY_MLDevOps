from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    directories =  [ 
        dataset_csv_path + "/ingestedfiles.txt" ,
        model_path + "/trainedmodel.pkl",
        model_path +"/latestscore.txt"
        ]

    for directory in directories:
        try:
            shutil.copy(directory, prod_deployment_path)
        except:
            raise Exception("missing file: "+ str(directory))        
        
if __name__ == '__main__':
    store_model_into_pickle()        

