'''
Train and test the model

Author: Sebastian Obando Morales
Date: June 20, 2022
'''

# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from joblib import dump,load

# Add the necessary imports for the starter code.
from ml.data import load_data,process_data
from ml.model import train_model, compute_model_metrics, inference, measure_model_performance
from ml.constants import path_name,path_encoder,path_lb,categorical_variables

# Add code to load in the data.
data = load_data(path_name)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Set categorical variables
cat_features = categorical_variables

# Proces the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Save encoder and lb
dump(encoder, path_encoder) 
dump(lb, path_lb) 

# Load encoder and lb
encoder = load(path_encoder)
lb = load(path_lb)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
train_model(X_train, y_train)

# Score the model
preds = inference(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# Outputs the performance of the model on slices of the data
categorical_variable = "occupation"
measure_model_performance(test, X_test, y_test, categorical_variable)
