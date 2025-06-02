import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup

# Ensure the path is correct
path_to_dataprocessor = Path('/home/oshadi/SISR-Final_Year_Project/envs/Project2/dataset').resolve()
if str(path_to_dataprocessor) not in sys.path:
    sys.path.insert(0, str(path_to_dataprocessor))

# import dataprocessor 
from dataprocessor import DataFrameProcessor, DataPreprocessor
from datavisualizer import DataVisualizer
from dataloaders import load_data 
# Load the data 
dfGeoCustomer, dfSFA_OrdersAll, dfSFA_Orders1, dfSFA_GPSData = load_data()


# URL of the Wikipedia page
url = 'https://en.wikipedia.org/wiki/Districts_of_Sri_Lanka'
# Send a GET request
response = requests.get(url)

# Parse the HTML content of the page with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table that contains the list of districts
#table = soup.find('table', {'class': 'wikitable sortable'})
# Find the table that contains the list of districts
table = soup.find('table', {'class': 'wikitable sortable static-row-numbers static-row-header-text'})

# Get all the rows in the table
rows = table.find_all('tr')

# List to store the names of the districts
districts = []

# Loop through the rows (skip the first one because it's the header row)
for row in rows[1:]:
    # Get the cells in the row
    cells = row.find_all('td')
    
    # Check if cells is not empty
    if cells:
        # Get the name of the district from the first cell
        district = cells[0].get_text(strip=True)
        
        # Add the district to the list
        districts.append(district)


DataFrameprocessor = DataFrameProcessor()
# Group by 'province' and count 'No'
dfGeoCustomer_ByDistrictList = DataFrameprocessor.group_and_manage(dfGeoCustomer, 'district', 'No', 'Count', operation='count')
# Group by 'province' and count 'No'
dfGeoCustomer_ByProvinceList = DataFrameprocessor.group_and_manage(dfGeoCustomer, 'province', 'No', 'Count', operation='count')
# Group by 'province' and apply list to 'No'
dfGeoCustomer_ByProvince = DataFrameprocessor.group_and_manage(dfGeoCustomer, 'province', 'No', 'List', operation='list')
# Get the districts in dfGeoCustomer_ByDistrictList

districts_in_df = dfGeoCustomer_ByDistrictList.index.tolist()
# Remove the second term from the districts in dfGeoCustomer_ByDistrictList
districts_in_df = [district.split(' ')[0] for district in districts_in_df]
# Remove the second term from the districts in dfGeoCustomer_ByDistrictList
districts = [district.split(' ')[0] for district in districts]

# Get the districts that are not in dfGeoCustomer_ByDistrictList
districts_not_in_df = [district for district in districts if district not in districts_in_df]

# Print the districts that are not in dfGeoCustomer_ByDistrictList
print('No Customers in these districts: ')
for district in districts_not_in_df:
    print(district)


