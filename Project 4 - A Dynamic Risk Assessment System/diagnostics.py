
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import sys
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path']) 
labels = config['labels']
features = config['features']

##################Function to get model predictions
def model_predictions(file_path):

    df = pd.read_csv(file_path)
    
    #read the deployed model and a test dataset, calculate predictions
    X = df.loc[:,features].values.reshape(-1, len(features))
    y = df[labels].values.reshape(-1, 1).ravel()
    ## Get model
    with open(prod_deployment_path + "/trainedmodel.pkl", 'rb') as file:
        model = pickle.load(file)
    ## Get predictions
    predicted = model.predict(X)    

    return predicted

##################Function to get summary statistics
def dataframe_summary(file_path):

    df = pd.read_csv(file_path)

    #calculate summary statistics here
    summary = []
    for feature in features:
        summary .append([feature, "mean", df[feature].mean()])
        summary .append([feature, "median", df[feature].median()])
        summary .append([feature, "standard deviation", df[feature].std()])
    
    return summary 

##################Function to evaluate missing data
def missing_data(file_path):
    
    df = pd.read_csv(file_path)

    missing_values = []

    for column in df.columns:
        count_na = df[column].isna().sum()
        count_not_na = df[column].count()
        count_total = count_not_na + count_na

        missing_values.append([column, str(int(count_na/count_total*100))+"%"])
    
    return str(missing_values)    

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    time_measure = []

    for procedure in ["training.py" , "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system('python3 %s' % procedure)
        timing=timeit.default_timer() - starttime
        time_measure.append([procedure, timing])
 
    return str(time_measure)

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    return str(outdated_packages)

if __name__ == '__main__':
    file_path = test_data_path + '/testdata.csv'
    model_predictions(file_path)
    dataframe_summary(file_path)
    missing_data(file_path)
    execution_time()
    outdated_packages_list()





    
