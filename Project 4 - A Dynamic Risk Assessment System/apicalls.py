import requests
import json
import os
# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Call each API endpoint and store the responses
response1 = requests.get(URL + '/prediction?file_path=testdata/testdata.csv').text
response2 = requests.get(URL + '/scoring').text
response3 = requests.get(URL + '/summarystats?file_path=testdata/testdata.csv').text
response4 = requests.get(URL + '/diagnostics?file_path=testdata/testdata.csv').text

# Combine all API responses
responses = ("##################### Response 1 #####################" + "\n" + response1 + "\n" + 
             "##################### Response 2 #####################" + "\n" + response2 + "\n" + 
             "##################### Response 3 #####################" + "\n" + response3 + "\n" + 
             "##################### Response 4 #####################" + "\n" + response4
             )

# Write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])

with open(os.path.join(model_path, "apireturns.txt"), "w") as returns_file:
    returns_file.write(responses)


