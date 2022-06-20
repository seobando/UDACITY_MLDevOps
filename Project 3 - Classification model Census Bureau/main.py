'''
Main module to run the pipeline

Author: Sebastian Obando Morales
Date: June 19, 2022
'''

# Put the code for your API here.
import logging
import pickle
import os
from typing import List
import pandas as pd

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from starter.ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system('rm -rf .dvc/cache')
    os.system('rm -rf .dvc/tmp/lock')
    os.system('dvc config core.hardlink_lock true')
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")

# load our model artifacts
root_path = os.path.dirname(os.path.realpath(__file__))
logger.info("Uploading model artifacts")
filename = '/gbm_model.pickle'
model = pickle.load(open(os.path.join(root_path, 'model') + filename, 'rb'))
filename = '/model_encoder.pickle'
encoder = pickle.load(open(os.path.join(root_path, 'model') + filename, 'rb'))
filename = '/label_encoder.pickle'
lb = pickle.load(open(os.path.join(root_path, 'model') + filename, 'rb'))


# Declare the data object with its components and their type.
class InputSchema(BaseModel):
    age: int
    workclass: str
    fnlgt: float
    education: str
    education_num: int = Field(None, alias="education-num")
    marital_status: str = Field(None, alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(None, alias="capital-gain")
    capital_loss: float = Field(None, alias="capital-loss")
    hours_per_week: int = Field(None, alias="hours-per-week")
    native_country: str = Field(None, alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 20,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 0,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0.0,
                "capital-loss": 0.0,
                "hours-per-week": 40.0,
                "native-country": "United-States"
            }
        }


class MultipleDataInputs(BaseModel):
    inputs: List[InputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "age": 20,
                        "workclass": "State-gov",
                        "fnlgt": 77516,
                        "education": "Bachelors",
                        "education-num": 13,
                        "marital-status": "Never-married",
                        "occupation": "Adm-clerical",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "Male",
                        "capital-gain": 2174,
                        "capital-loss": 0,
                        "hours-per-week": 40,
                        "native-country": "United-States"
                    }
                ]
            }
        }


# Instantiate the app.
app = FastAPI()


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hi, this is the Welcome message for the api!"}


@app.post("/predict")
async def predict(item: InputSchema):
    # Convert input data into a dictionary
    # https://stackoverflow.com/questions/71474569/response-500-dict-to-pandas-dataframe
    # Convert the dictionary into a dataframe
    df_pred = pd.DataFrame(jsonable_encoder(item), index=["value"])
    data_pred, _, _, _ = process_data(
        df_pred, categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"], label=None, training=False, encoder=encoder, lb=lb)

    predictions = '<=50k' if str(model.predict(data_pred)[0]) == '0' else '>50k'
    return {"prediction": predictions}