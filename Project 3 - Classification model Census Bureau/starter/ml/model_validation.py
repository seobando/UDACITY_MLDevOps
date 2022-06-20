import requests


if __name__ == "__main__":
    #https://knowledge.udacity.com/questions/831266
    data = { "age": 20, "workclass": "State-gov", "fnlgt": 77516, "education": "Bachelors", "education-num": 0, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 0.0, "capital-loss": 0.0, "hours-per-week": 40, "native-country": "United-States"}

    response = requests.post(url='http://127.0.0.1:8000/predict', json=data)

    print(response.status_code)

    print(response.json())

