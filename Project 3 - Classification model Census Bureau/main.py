'''
Main module to run the pipeline

Author: Sebastian Obando Morales
Date: June 19, 2022
'''

from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
from joblib import load

from starter.ml.data import process_data
from starter.ml.constants import categorical_variables,path_encoder,path_lb

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
        item.marital_status,
        item.occupation,
        item.relationship,
        item.race,
        item.sex,
        item.hours_per_week,
        item.native_country
    ]])

    data_input = pd.DataFrame(data = input_array, columns =[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours_per_week",
        "native-country"
    ])

    encoder = load("./model/encoder.joblib")
    lb = load("./model/lb.joblib")

    X_input, _, _, _ = process_data(
        data_input, categorical_features=categorical_variables, label=None, training=False, encoder=encoder, lb=lb
    )

    preds = inference(X_input)

    print(preds)

    return {"prediction" : preds[0]}