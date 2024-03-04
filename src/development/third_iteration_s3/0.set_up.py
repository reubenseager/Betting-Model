"""
This script is here to set up the project. I will be importing the required packages, setting the directories and creating the aws s3 storage.    

"""

#Imports
#File Management
import os
from pathlib import Path
import glob

#Data Manipulation
import pandas as pd
import numpy as np # Package for scientific computing
from functools import reduce  # Used to merge multiple dataframes together


#Date and Time Manipulation
from datetime import datetime, timedelta
from dateutil.parser import parse
from dateutil import parser
import pytz

#Web-Sraping
import requests # Used to access and download information from websites
from bs4 import BeautifulSoup # Package for working with html and information parsed using requests
import time # Package to slow down the webscraping process
import io # Package to convert the API data to a pandas dataframe

#Storage
import pyarrow.feather as feather   # Package to store dataframes in a binary format
import s3fs #Package to store data in s3
import boto3 #Package to interact with s3
import awswrangler as wr #Package to interact with s3
import joblib #Package to save python objects as binary files


#Misceallaneous
from tqdm import tqdm # Package to show progress bar

#Testing imports
import pytest # Package for testing code


#Setting the working directory. This will be used to access python files but the data will be stored in s3
os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")

#Creating S3 storage

#Using S3 client over resource as it is a lower level interfacwe
s3_client = boto3.client('s3', region_name='us-east-1')

s3_bucket_name = "epl-prediction-model-data"

#Creating a new bucket
respone = s3_client.create_bucket(Bucket=s3_bucket_name)

#Creating folders within the bucket (not really subfolders but it's effectively how you can think about it)

#Data folders and primary subfolders
s3_client.put_object(Bucket=s3_bucket_name, Key="data/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/raw/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/intermediate/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/output/")

#Model folders and subfolders
s3_client.put_object(Bucket=s3_bucket_name, Key="models/")
s3_client.put_object(Bucket=s3_bucket_name, Key="models/model_studies/")
s3_client.put_object(Bucket=s3_bucket_name, Key="models/inference_model/")


#Secondary subfolders
s3_client.put_object(Bucket=s3_bucket_name, Key="data/raw/historical_betting_odds/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/raw/team_name_lookup/")

s3_client.put_object(Bucket=s3_bucket_name, Key="data/intermediate/elo_data/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/intermediate/historical_betting_odds/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/intermediate/live_betting_odds/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/intermediate/webscraped_football_data/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/intermediate/webscraped_fifa_data/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/intermediate/combined_betting_data/")
s3_client.put_object(Bucket=s3_bucket_name, Key="data/intermediate/pre_processed/")


#Global variables
current_season = 2024 #The current season that we are in
first_season = 2014 #The first season that we are looking at
window_size = 3 #The window size for the moving average
#Running the script below
#1. Webscraping Premier League Data
from webscraping_epl_data_function import webscraping_epl_data_function
webscraping_epl_data_function(first_season = first_season, last_season = current_season, reread_data = False, s3_bucket_name = s3_bucket_name)

#2. Dowbloading ELO Data
from elo_ratings_api_function import elo_ratings_api_function
elo_ratings_api_function(s3_bucket_name = s3_bucket_name)

#3. Webscraping Fifa Ratings
from webscraping_fifa_ratings_function import webscraping_fifa_ratings_function
webscraping_fifa_ratings_function(reread_data = False, s3_bucket_name = "epl-prediction-model-data")

#4. Downloading Historical Betting Odds
from historical_betting_odds_function import historical_betting_odds_function
historical_betting_odds_function(last_season = 2324, reread_data = False, s3_bucket_name = s3_bucket_name)

#5. Downloading Live Betting Odds
from live_odds_api_function import live_betting_odds_function
live_betting_odds_function(s3_bucket_name=s3_bucket_name)

#6. Preprocessing the data
from pre_processing_data_function import pre_processing_function
pre_processing_function(s3_bucket_name = s3_bucket_name, window_size=window_size)


