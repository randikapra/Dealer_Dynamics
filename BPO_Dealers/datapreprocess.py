
# # Define the columns to keep
# columns = ['Date', 'DistributorCode', 'UserCode', 'FinalValue']

# # Create copies and preprocess the DataFrames
# dfSFA_OrdersJanCopy = DataPreprocessor.copy_and_preprocess(dfSFA_OrdersJan, columns)
# dfSFA_OrdersFebCopy = DataPreprocessor.copy_and_preprocess(dfSFA_OrdersFeb, columns)
# dfSFA_OrdersMarCopy = DataPreprocessor.copy_and_preprocess(dfSFA_OrdersMar, columns)
# dfSFA_Orders1Copy = DataPreprocessor.copy_and_preprocess(dfSFA_Orders1, columns)
# dfSFA_OrdersAllCopy = DataPreprocessor.copy_and_preprocess(dfSFA_OrdersAll, columns)

# # Create a copy, drop the 'Unnamed: 0' column from dfGeoCustomer and convert the 'No' column to string
# dfGeoCustomer = DataPreprocessor.copy_and_drop(dfGeoCustomer, ['Unnamed: 0'])
# dfGeoCustomer['No'] = dfGeoCustomer['No'].astype(str)

# # dfSFA_GPSData.info() #January
# # dfSFA_GPSData.dropna()



# dfSFA_OrdersJanCopy = DataPreprocessor.merge_dataframes(dfSFA_OrdersJanCopy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')
# dfSFA_OrdersFebCopy = DataPreprocessor.merge_dataframes(dfSFA_OrdersFebCopy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')
# dfSFA_OrdersMarCopy = DataPreprocessor.merge_dataframes(dfSFA_OrdersMarCopy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')
# dfSFA_Orders1Copy = DataPreprocessor.merge_dataframes(dfSFA_Orders1Copy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')
# dfSFA_OrdersAllCopy = DataPreprocessor.merge_dataframes(dfSFA_OrdersAllCopy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')


# dfSFA_OrdersJanCopy


