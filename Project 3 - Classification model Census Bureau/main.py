'''
Main module to run the pipeline

Author: Sebastian Obasampndo Morales
Date: June 27, 2022
'''

from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
from joblib import load

from starter.ml.data import process_data
from starter.ml.constants import categorical_variables,path_encoder,path_lb
import starter.ml.model

import logging

class TaggedItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

app = FastAPI()

@app.get("/")
async def get_items():
    return {"message" : "This is greeting page of applications !"}

@app.post("/")
async def inference(item: TaggedItem):
    input_array = np.array([[
        item.age,
        item.workclass,
        item.fnlgt,
        item.education,
        item.education_num,
        item.marital_status,
        item.occupation,
        item.relationship,
        item.race,
        item.sex,
        item.capital_gain,
        item.capital_loss,
        item.hours_per_week,
        item.native_country
    ]])

    data_input = pd.DataFrame(data = input_array, columns =[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num", 
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain", 
        "capital-loss",         
        "hours_per_week",
        "native-country"
    ])

    encoder = load(path_encoder)
    lb = load(path_lb)

    X_input, _, _, _ = process_data(
        data_input, categorical_features=categorical_variables, label=None, training=False, encoder=encoder, lb=lb
    )

    pred = starter.ml.model.inference(X_input)
    y = lb.inverse_transform(pred)[0]
   
    return {"prediction" : y}