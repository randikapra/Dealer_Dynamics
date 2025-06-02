
DataFrameprocessor = DataFrameProcessor()
grouped_df_FinalSale = DataFrameprocessor.group_and_calculate(dfSFA_OrdersJanCopy, 'DistributorCode', {'FinalValue': 'sum', 'Latitude': 'first', 'Longitude': 'first'}, 'FinalValue_Millions (MM)', 1e6)
print('Customers with sales in January month ::')
DataFrameprocessor.print_dataframe(grouped_df_FinalSale, ['DistributorCode','FinalValue_Millions (MM)'])
print('Same total sales in a month with different Dealers')
repeating_finalvalue_unique_latitude = DataFrameprocessor.find_repeating_values(grouped_df_FinalSale, ['FinalValue', 'Latitude'])
DataFrameprocessor.print_dataframe(repeating_finalvalue_unique_latitude, ['DistributorCode', 'FinalValue', 'FinalValue_Millions (MM)'])

