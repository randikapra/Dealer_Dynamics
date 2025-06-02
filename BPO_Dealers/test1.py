# # import sys
# # import os

# # # Ensure the path is correct
# # path_to_dataprocessor = '/home/oshadi/SISR-Final_Year_Project/envs/Project2/dataset'
# # if path_to_dataprocessor not in sys.path:
# #     sys.path.insert(0, path_to_dataprocessor)
# #     print('hi')

# # try:
# #     from dataprocessor import DataFrameProcessor
# # except ModuleNotFoundError as e:
# #     print(f"Error: {e}. Please ensure the module path and dependencies are correct.")



import sys
from pathlib import Path

# Ensure the path is correct
path_to_dataprocessor = Path('/home/oshadi/SISR-Final_Year_Project/envs/Project2/dataset').resolve()
if str(path_to_dataprocessor) not in sys.path:
    sys.path.insert(0, str(path_to_dataprocessor))
    print('Path added to sys.path')

try:
    from dataprocessor import DataFrameProcessor
    print('Module imported successfully')
except ModuleNotFoundError as e:
    print(f"Error: {e}. Please ensure the module path and dependencies are correct.")
    print(f"Current sys.path: {sys.path}")



import sys 
from pathlib import Path 
# Add the directory containing dataprocessor.py to the system path 
dataprocessor_path = Path('/home/oshadi/SISR-Final_Year_Project/envs/Project2/dataset').resolve() 
sys.path.append(str(dataprocessor_path)) 
# Now you can import dataprocessor 
from dataprocessor import DataFrameProcessor 
# Your existing imports 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import requests 
from bs4 import BeautifulSoup 
# URL of the Wikipedia page 
url = 'https://en.wikipedia.org/wiki/Districts_of_Sri_Lanka' 
assert dataprocessor_path.exists(), f"Path {dataprocessor_path} does not exist."
