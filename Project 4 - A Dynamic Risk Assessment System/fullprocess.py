

import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os

from ingestion import merge_multiple_dataframe

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path']) 

##################Check and read new data
#first, read ingestedfiles.txt
ingested_files =[]
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as report_file:
    for line in report_file:
        ingested_files.append(line.rstrip())
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files = False
for filename in os.listdir(input_folder_path):
    if input_folder_path + "/" + filename not in ingested_files:
        new_files = True


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not new_files:
    print("No new ingested data, exiting")
    exit(0)

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as report_file:
    old_f1 = float(report_file.read())

with open(os.path.join(model_path, "latestscore.txt"), "r") as report_file:
    new_f1 = float(report_file.read())

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_f1 >= old_f1:
    print("Actual F1 (%s) is better/equal than old F1 (%s), no drift detected -> exiting" % (new_f1, old_f1))    
    exit(0)
else:
    print("Actual F1 (%s) is WORSE than old F1 (%s), drift detected -> training model" % (new_f1, old_f1)) 

##################Re-training
os.system('python training.py')
os.system('python scoring.py')

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
os.system('python deployment.py')
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python apicalls.py')
os.system('python reporting.py')






