from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import model_predictions, dataframe_summary, missing_data, outdated_packages_list, execution_time
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prediction_model = None

#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    dataset_path =  os.getcwd() + '/' + request.args.get("file_path")
    preds = model_predictions(dataset_path)
    return jsonify({"prediction": preds.tolist()})

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    return jsonify({"f1_score": score_model()})

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    dataset_path = os.getcwd() + '/' + request.args.get("file_path")
    return  jsonify({"stats": dataframe_summary(dataset_path)})

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    dataset_path =  os.getcwd() + '/' + request.args.get("file_path")
    return jsonify({
        "timing": execution_time(),
        "missing_data": missing_data(dataset_path),
        "dependency_check": outdated_packages_list(),
    })

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
