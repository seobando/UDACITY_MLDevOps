import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file    
    ## Declare variables
    input_directory = os.getcwd() + '/' + input_folder_path
    output_directory = os.getcwd() + '/' + output_folder_path
    file_names = os.listdir(input_directory)
    final_dataframe = pd.DataFrame(columns=['corporation','lastmonth_activity','lastyear_activity','number_of_employees','exited'])
    ## Consolidate data
    for file_name in file_names:
        if ".csv" in file_name:
            file_path = input_directory +'/'+ file_name
            current_df = pd.read_csv(file_path)
            final_dataframe = final_dataframe.append(current_df).reset_index(drop=True)    
    ## Remove duplications
    final_dataframe = final_dataframe.drop_duplicates()
    ## Save the data consolidated
    final_dataframe.to_csv(output_directory + '/finaldata.csv', index=False)        
    ## Save the report of dataframes used in the consolidation
    with open( output_directory + "/ingestedfiles.txt",'w') as report_file:
        for file_name in file_names:
            if ".csv" in file_name:
                report_file.write(str(file_name) + "\n")           

if __name__ == '__main__':
    merge_multiple_dataframe()
