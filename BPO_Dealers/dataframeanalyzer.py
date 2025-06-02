
analyzerDFA = DataFrameAnalyzer()
analyzerDFA.print_unique_counts(dfSFA_OrdersJanCopy, ['DistributorCode', 'UserCode'])
analyzerDFA.print_total_sales(dfSFA_OrdersJanCopy, 1)
analyzerDFA.print_top_contributors(dfSFA_OrdersJanCopy, 'DistributorCode', 'FinalValue')
analyzerDFA.print_info(dfSFA_OrdersJanCopy)
analyzerDFA.print_info(dfSFA_OrdersAllCopy)
analyzerDFA.print_info(dfSFA_Orders1Copy)
analyzerDFA.print_info(dfGeoCustomer)
analyzerDFA.print_fraction_of_sales_no_location(dfSFA_OrdersJanCopy, 1)

