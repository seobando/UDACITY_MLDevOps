import requests

data = {
    "age": 32,
    "workclass": "Private",
    "fnlgt": 0,
    "education": "Some-college",
    "education_num": 0,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 60,
    "native_country": "United-States"
    } 

# response = requests.post(url='http://127.0.0.1:8000/', json=data, headers={"Content-Type": "application/json; charset=utf-8"})
response = requests.post(
    url='http://app-udacity.herokuapp.com/', 
    json=data, 
    headers={"Content-Type": "application/json; charset=utf-8"}
    )

print(response.status_code)
print(response.reason)
print(response.json())