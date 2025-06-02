# import sys
# from pathlib import Path
# import matplotlib.pyplot as plt
# import numpy as np

# # Ensure the path is correct
# path_to_dataprocessor = Path('/home/oshadi/SISR-Final_Year_Project/envs/Project2/dataset').resolve()
# if str(path_to_dataprocessor) not in sys.path:
#     sys.path.insert(0, str(path_to_dataprocessor))

# # import dataprocessor 
# from dataprocessor import DataFrameProcessor, DataPreprocessor
# from datavisualizer import DataVisualizer
# from dataloaders import load_data 
# # Load the data 
# dfGeoCustomer, dfSFA_OrdersAll, dfSFA_Orders1, dfSFA_GPSData = load_data()
# DataFrameprocessor = DataFrameProcessor()
# # Group by 'province' and count 'No'
# dfGeoCustomer_ByDistrictList = DataFrameprocessor.group_and_manage(dfGeoCustomer, 'district', 'No', 'Count', operation='count')
# # Group by 'province' and count 'No'
# dfGeoCustomer_ByProvinceList = DataFrameprocessor.group_and_manage(dfGeoCustomer, 'province', 'No', 'Count', operation='count')
# # Group by 'province' and apply list to 'No'
# dfGeoCustomer_ByProvince = DataFrameprocessor.group_and_manage(dfGeoCustomer, 'province', 'No', 'List', operation='list')
# # Get the districts in dfGeoCustomer_ByDistrictList

# districts_in_df = dfGeoCustomer_ByDistrictList.index.tolist()

# DataVisualizer.visualize_bar(dfGeoCustomer_ByProvinceList, 
#                                 'Number of Dealers by Province', 
#                                 'Province', 
#                                 'Number of Dealers')
# plt.figure(figsize=(10,6))
# bars = plt.bar(range(len(dfGeoCustomer_ByProvinceList)), dfGeoCustomer_ByProvinceList)
# plt.title('Dealers in each Province')
# plt.xlabel('Province')
# plt.ylabel('Number of Dealers')

# # Get the heights of the bars
# heights = [bar.get_height() for bar in bars]

# # Color the max height bar as red and min height bar as green
# bars[np.argmax(heights)].set_color('r')
# bars[np.argmin(heights)].set_color('g')

# plt.show()

# DataVisualizer.visualize_pie(dfGeoCustomer_ByProvinceList.values, 
#                              dfGeoCustomer_ByProvinceList.index, 
#                              'Number of Dealers by Province'
#                              # autopct='%1.1f%%'
#                             )


# DataVisualizer.visualize_bar(dfGeoCustomer_ByDistrictList, 
#                                 'Number of Dealers by District', 
#                                 'District', 
#                                 'Number of Dealers')
# DataVisualizer.visualize_pie(dfGeoCustomer_ByDistrictList.values, 
#                              dfGeoCustomer_ByDistrictList.index, 
#                              'Number of Dealers by District',
#                             )
# dfGeoCustomer_ByCityList = dfGeoCustomer.groupby('city')['No'].count().sort_values()
# #dfGeoCustomer_ByCity = dfGeoCustomer.groupby('city')['No'].apply(list)
# dfGeoCustomer_ByCityList
# colormaps = [
#     ["Perceptually Uniform Sequential", 
#         ['viridis', 'plasma', 'inferno', 'magma', 'cividis']],
#     ["Sequential", 
#          ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']],
#     ["Sequential (2)", 
#          ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink']],
#     ["Diverging", 
#          ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu']],
#     ["Cyclic", 
#          ['twilight', 'twilight_shifted', 'hsv']],
#     ["Qualitative", 
#          ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1']],
#     ["Miscellaneous", 
#          ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern']]
# ]

# DataVisualizer.visualize_pie(dfGeoCustomer_ByProvinceList.values, 
#                              dfGeoCustomer_ByProvinceList.index, 
#                              'Number of Dealers by Province'
#                              # autopct='%1.1f%%'
#                             )
# # Visualize the distribution of districts within each province
# DataVisualizer.visualize_nested_pie(dfGeoCustomer, 'province', 'district', 'District distribution in each Province')

# DataVisualizer.visualize_bar(dfGeoCustomer_ByProvinceList, 'Dealers in each Province', 'Province', 'Number of Dealers')
# DataVisualizer.visualize_pie(dfGeoCustomer_ByProvinceList.values.tolist(), dfGeoCustomer_ByProvinceList.index.tolist(), 'Dealers by Province', autopct='%1.1f%%')
# DataVisualizer.visualize_bar(dfGeoCustomer_ByDistrictList, 'Dealers in each District', 'District', 'Number of Dealers')
# DataVisualizer.visualize_pie(dfGeoCustomer_ByDistrictList.values.tolist(), dfGeoCustomer_ByDistrictList.index.tolist(), 'Dealers by District')
# DataVisualizer.visualize_pie(dfGeoCustomer['province'].value_counts(), dfGeoCustomer['province'].value_counts().index.tolist(), 'Province distribution', autopct='%1.1f%%')
# DataVisualizer.visualize_nested_pie(dfGeoCustomer, 'province', 'district', 'District distribution')




# top_distributors = dfSFA_Orders1.groupby('DistributorCode')['FinalValue'].sum().sort_values(ascending=False).head(20)
# DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_distributors, 'Top 20 Dealers in January Orders','Distributor','Value (millions))', 'bar' )
# DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_distributors, 'Top 20 Dealers in January Orders','Distributor','Value (millions))',  'pie' )


# # top 20 bpos contribute to sales
# top_users = dfSFA_Orders1.groupby('UserCode')['FinalValue'].sum().sort_values(ascending=False)
# DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_users, 'BPOs in January Orders','BPO', 'Value', 'bar' )
# DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_users, 'BPOs in January Orders','BPO', 'Value', 'pie' )

# # top 10 bpos contribute to sales
# DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_users.head(10) , 'Top 10 BPOs in January Orders','BPO', 'Value', 'pie' )


# DataVisualizer.plot_time_series(dfSFA_Orders1, 'Date', 'FinalValue', 'Jan Orders', 'Total Sales Distribution Over Time')

# # analyzerDFA.print_fraction_of_sales_no_location(dfSFA_Orders1, 1)

# # visualize_dataframe(dfSFA_OrdersJanCopy, null_rowsAllRep[['DistributorCode', 'FinalValue']], 'Dealers with no Geometric Informations', 'bar' )


# dfSFA_OrdersJanCopy = DataPreprocessor.dropna(dfSFA_Orders1, ['Latitude', 'Longitude'])


import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Assuming this is needed for DataFrame operations
from dataprocessor import DataFrameProcessor, DataPreprocessor
from datavisualizer import DataVisualizer
from dataloaders import load_data

def add_custom_paths():
    """
    Adds custom paths to sys.path to ensure modules are found.
    """
    path_to_dataprocessor = Path('/home/oshadi/SISR-Final_Year_Project/envs/Project2/dataset').resolve()
    if str(path_to_dataprocessor) not in sys.path:
        sys.path.insert(0, str(path_to_dataprocessor))

def process_data():
    """
    Loads and processes the data using defined functions.
    
    Returns:
        tuple: Dataframes with processed data.
    """
    dfGeoCustomer, dfSFA_OrdersAll, dfSFA_Orders1, dfSFA_GPSData = load_data()
    dfGeoCustomer_ByDistrictList = DataFrameProcessor().group_and_manage(dfGeoCustomer, 'district', 'No', 'Count', operation='count')
    dfGeoCustomer_ByProvinceList = DataFrameProcessor().group_and_manage(dfGeoCustomer, 'province', 'No', 'Count', operation='count')
    dfGeoCustomer_ByProvince = DataFrameProcessor().group_and_manage(dfGeoCustomer, 'province', 'No', 'List', operation='list')
    return dfGeoCustomer, dfGeoCustomer_ByDistrictList, dfGeoCustomer_ByProvinceList, dfSFA_Orders1

def visualize_data(dfGeoCustomer_ByDistrictList, dfGeoCustomer_ByProvinceList, dfSFA_Orders1):
    """
    Visualizes the processed data using different visualization techniques.
    """
    DataVisualizer.visualize_bar(dfGeoCustomer_ByProvinceList, 'Number of Dealers by Province', 'Province', 'Number of Dealers')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(dfGeoCustomer_ByProvinceList)), dfGeoCustomer_ByProvinceList)
    plt.title('Dealers in each Province')
    plt.xlabel('Province')
    plt.ylabel('Number of Dealers')

    heights = [bar.get_height() for bar in bars]
    bars[np.argmax(heights)].set_color('r')
    bars[np.argmin(heights)].set_color('g')
    plt.show()

    DataVisualizer.visualize_pie(dfGeoCustomer_ByProvinceList.values, dfGeoCustomer_ByProvinceList.index, 'Number of Dealers by Province')
    DataVisualizer.visualize_bar(dfGeoCustomer_ByDistrictList, 'Number of Dealers by District', 'District', 'Number of Dealers')
    DataVisualizer.visualize_pie(dfGeoCustomer_ByDistrictList.values, dfGeoCustomer_ByDistrictList.index, 'Number of Dealers by District')

    top_distributors = dfSFA_Orders1.groupby('DistributorCode')['FinalValue'].sum().sort_values(ascending=False).head(20)
    DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_distributors, 'Top 20 Dealers in January Orders','Distributor','Value (millions))', 'bar')
    DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_distributors, 'Top 20 Dealers in January Orders','Distributor','Value (millions))', 'pie')

    top_users = dfSFA_Orders1.groupby('UserCode')['FinalValue'].sum().sort_values(ascending=False)
    DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_users, 'BPOs in January Orders','BPO', 'Value', 'bar')
    DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_users, 'BPOs in January Orders','BPO', 'Value', 'pie')
    DataVisualizer.visualize_dataframe(dfSFA_Orders1, top_users.head(10), 'Top 10 BPOs in January Orders','BPO', 'Value', 'pie')

    DataVisualizer.plot_time_series(dfSFA_Orders1, 'Date', 'FinalValue', 'Jan Orders', 'Total Sales Distribution Over Time')
    
    dfSFA_OrdersJanCopy = DataPreprocessor().dropna(dfSFA_Orders1, ['Latitude', 'Longitude'])

def main():
    """
    Main function to run the data processing and visualization.
    """
    add_custom_paths()
    dfGeoCustomer, dfGeoCustomer_ByDistrictList, dfGeoCustomer_ByProvinceList, dfSFA_Orders1 = process_data()
    visualize_data(dfGeoCustomer_ByDistrictList, dfGeoCustomer_ByProvinceList, dfSFA_Orders1)

if __name__ == "__main__":
    main()
