
'''
Test cases for the sanity_check

Author: Sebastian Obasampndo Morales
Date: June 28, 2022
'''

from fastapi.testclient import TestClient
import logging
import pytest
from main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

client = TestClient(app)

@pytest.fixture
def client():
    """
    Get client
    """
    api_client = TestClient(app)
    return api_client


def test_read_main(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "This is greeting page of applications !"}


def test_get_prediction_negative(client):
    data_test = {
        "age": 20,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 0,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post('/', json=data_test)
    print(f"status_code :{response.status_code}")
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_get_prediction_positive(client):
    data_test = {
        "age": 53,
        "workclass": "Private",
        "fnlgt": 123011,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
    response = client.post('/', json=data_test)
    print(f"status_code :{response.status_code}")
    assert response.status_code == 200, response.json()
    assert response.json() == {"prediction": ">50K"}