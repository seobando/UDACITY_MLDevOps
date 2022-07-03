
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    with open(prod_deployment_path/"trainedmodel.pkl", 'rb') as f:
        pipeline = pickle.load(f)
    preds = pipeline.predict(dataset)
    return preds.tolist()

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    finaldata = pd.read_csv(dataset_csv_path/"finaldata.csv", low_memory=False)
    stats_list = []

    for col in finaldata.select_dtypes(include=np.number).columns:
        stats_list.append(np.mean(finaldata[col]))
        stats_list.append(np.median(finaldata[col]))
        stats_list.append(np.std(finaldata[col]))
    
    # convert np.float64 to float
    stats_list = [float(x) for x in stats_list]
    return stats_list

##################Function to evaluate missing data
def missing_data():
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    result = []
    for column in df.columns:
        count_na = df[column].isna().sum()
        count_not_na = df[column].count()
        count_total = count_not_na + count_na

        result.append([column, str(int(count_na/count_total*100))+"%"])
    
    return str(result)    

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    delay_list = []

    t_1 = timeit.default_timer()
    os.system('python3 ingestion.py')
    dt = timeit.default_timer() - t_1
    delay_list.append(dt)

    t_1 = timeit.default_timer()
    os.system('python3 training.py')
    dt = timeit.default_timer() - t_1
    delay_list.append(dt)

    return delay_list

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    return str(outdated_packages)

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
