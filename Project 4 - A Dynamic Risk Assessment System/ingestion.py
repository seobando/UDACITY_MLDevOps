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

    # Part I - A
    directories=['/udacity1/','/udacity2/']

    final_dataframe = pd.DataFrame(columns=['peratio','price'])

    for directory in directories:
       filenames = os.listdir(os.getcwd()+directory)
       for each_filename in filenames:
           currentdf = pd.read_csv(os.getcwd()+directory+each_filename)
           final_dataframe=final_dataframe.append(currentdf).reset_index(drop=True)    

    final_dataframe.to_csv('demo_20210330.csv')

    # Part I - B
    directories=['/data1/','/data2/','/data3/']
    df_list = pd.DataFrame(columns=['col1','col2','col3'])

    for directory in directories:
        filenames = os.listdir(os.getcwd()+directory)

        for each_filename in filenames:
            df1 = pd.read_csv(os.getcwd()+directory+each_filename)
            df_list=df_list.append(df1)

    result=df_list.drop_duplicates()
    result.to_csv('result.csv', index=False)        

    # Part II
    sourcelocation='./recorddatasource/'
    filename='recordkeepingdemo.csv'
    outputlocation='records.txt'

    data=pd.read_csv(sourcelocation+filename)

    dateTimeObj=datetime.now()
    thetimenow=str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)

    allrecords=[sourcelocation,filename,len(data.index),thetimenow]
    
    MyFile=open(outputlocation,'w')
    for element in allrecords:
        MyFile.write(str(element))       

if __name__ == '__main__':
    merge_multiple_dataframe()
