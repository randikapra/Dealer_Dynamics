# from pathlib import Path 
# import os
# import pandas as pd 
# # Define the base directory path using Pathlib 
# # base_dir = Path('/home/oshadi/SISR-Final_Year_Project/envs/Project2/dataset/data')

# # Define the base directory path using an environment variable
# base_dir = os.getenv('DATA_PATH', '/home/oshadi/SISR-Final_Year_Project/envs/Project2/dataset/data')

# # Construct file paths using the base directory
# geo_customer_path = os.path.join(base_dir, 'AllGeo (1).csv')
# sfa_orders_path = os.path.join(base_dir, 'SFA_Orders_202.xlsx')
# gps_data_path = os.path.join(base_dir, 'Given Datasets/SFA_GPSData_202_2023January.csv')

# # Read the data
# dfGeoCustomer = pd.read_csv(geo_customer_path)
# dfSFA_Orders = pd.read_excel(sfa_orders_path)
# dfSFA_GPSDataJan = pd.read_csv(gps_data_path)

# # Get all sheet names
# SFA_Orders_xls = pd.ExcelFile(sfa_orders_path)
# sheet_names = SFA_Orders_xls.sheet_names

# # Read each sheet into a separate DataFrame
# dfSFA_Orders = {sheet: pd.read_excel(SFA_Orders_xls, sheet_name=sheet) for sheet in sheet_names}

# dfSFA_OrdersJan = dfSFA_Orders['Jan']
# dfSFA_OrdersFeb = dfSFA_Orders['Feb']
# dfSFA_OrdersMar = dfSFA_Orders['Mar']
# dfSFA_OrdersJan.head()

# dfSFA_OrdersAll = pd.concat([dfSFA_OrdersJan, dfSFA_OrdersFeb, dfSFA_OrdersMar])
# dfSFA_Orders1 = pd.concat([dfSFA_OrdersJan, dfSFA_OrdersFeb])
# dfSFA_GPSData = dfSFA_GPSDataJan.copy()

from pathlib import Path
import os
import pandas as pd

def load_data():
    # Define the base directory path using Pathlib
    base_dir = Path(os.getenv('DATA_PATH', '/home/oshadi/SISR-Final_Year_Project/envs/Project2/dataset/data')).resolve()

    # Construct file paths using the base directory
    geo_customer_path = base_dir / 'AllGeo (1).csv'
    sfa_orders_path = base_dir / 'SFA_Orders_202.xlsx'
    gps_data_path = base_dir / 'Given Datasets/SFA_GPSData_202_2023January.csv'

    # Read the data
    dfGeoCustomer = pd.read_csv(geo_customer_path)
    dfSFA_Orders = pd.read_excel(sfa_orders_path)
    dfSFA_GPSDataJan = pd.read_csv(gps_data_path)

    # Get all sheet names
    SFA_Orders_xls = pd.ExcelFile(sfa_orders_path)
    sheet_names = SFA_Orders_xls.sheet_names

    # Read each sheet into a separate DataFrame
    dfSFA_Orders = {sheet: pd.read_excel(SFA_Orders_xls, sheet_name=sheet) for sheet in sheet_names}

    dfSFA_OrdersJan = dfSFA_Orders['Jan']
    dfSFA_OrdersFeb = dfSFA_Orders['Feb']
    dfSFA_OrdersMar = dfSFA_Orders['Mar']

    dfSFA_OrdersAll = pd.concat([dfSFA_OrdersJan, dfSFA_OrdersFeb, dfSFA_OrdersMar])
    dfSFA_Orders1 = pd.concat([dfSFA_OrdersJan, dfSFA_OrdersFeb])
    dfSFA_GPSData = dfSFA_GPSDataJan.copy()

    return dfGeoCustomer, dfSFA_OrdersAll, dfSFA_Orders1, dfSFA_GPSData

# # Test the function
# if __name__ == "__main__":
#     data = load_data()
#     for df in data:
#         print(df.head())
