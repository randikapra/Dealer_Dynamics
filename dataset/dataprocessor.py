import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import warnings

class DataPreprocessor:
    """
    A class used to preprocess data.

    ...

    Static Methods
    -------
    preprocess(df, columns) -> DataFrame
        Preprocesses the DataFrame by selecting columns and converting 'DistributorCode' to string.
    drop(df, columns) -> DataFrame
        Drops the specified columns from the DataFrame.
    copy_and_preprocess(df, columns) -> DataFrame
        Creates a copy of the DataFrame and preprocesses it.
    copy_and_drop(df, columns) -> DataFrame
        Creates a copy of the DataFrame and drops the specified columns.
    merge_dataframes(df1, df2, **kwargs) -> DataFrame
        Merges two DataFrames based on the specified keyword arguments.
    """

    @staticmethod
    def preprocess(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Preprocesses the DataFrame by selecting columns, converting 'DistributorCode' to string,
        and converting 'RecievedDate' to datetime format and extracting the date part.

        Parameters:
        df (pd.DataFrame): The DataFrame to preprocess.
        columns (list): The columns to select.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        try:
            df = df.loc[:, columns]
            df.loc[:, 'DistributorCode'] = df['DistributorCode'].astype(str)

            # Convert 'RecievedDate' to datetime format and extract the date part
            # df['Date'] = pd.to_datetime(df['Date'])
            # df['DateOnly'] = df['Date'].dt.date

            return df
        except Exception as e:
            logging.error("Error in preprocess method: %s", str(e))
            raise


    @staticmethod
    def drop(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to drop columns from.
        columns (list): The columns to drop.

        Returns:
        pd.DataFrame: The DataFrame with the columns dropped.
        """
        try:
            columns_to_drop = [col for col in columns if col in df.columns]
            return df.drop(columns=columns_to_drop)
        except Exception as e:
            logging.error("Error in drop method: %s", str(e))
            raise

    @staticmethod
    def copy_and_preprocess(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Creates a copy of the DataFrame and preprocesses it.

        Parameters:
        df (pd.DataFrame): The DataFrame to copy and preprocess.
        columns (list): The columns to select in the preprocessing step.

        Returns:
        pd.DataFrame: The copied and preprocessed DataFrame.
        """
        return DataPreprocessor.preprocess(df.copy(), columns)

    @staticmethod
    def copy_and_drop(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Creates a copy of the DataFrame and drops the specified columns.

        Parameters:
        df (pd.DataFrame): The DataFrame to copy and drop columns from.
        columns (list): The columns to drop.

        Returns:
        pd.DataFrame: The copied DataFrame with the columns dropped.
        """
        return DataPreprocessor.drop(df.copy(), columns)

    @staticmethod
    def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Merges two DataFrames based on the specified keyword arguments.

        Parameters:
        df1 (pd.DataFrame): The first DataFrame to merge.
        df2 (pd.DataFrame): The second DataFrame to merge.
        **kwargs: Keyword arguments to pass to the pandas merge function. Can optionally include a 'drop_column' argument to specify a column to drop from the merged DataFrame.

        Returns:
        pd.DataFrame: The merged DataFrame.
        """
        try:
            drop_column = kwargs.pop('drop_column', None)
            merged_df = df1.merge(df2, **kwargs)
            if drop_column:
                merged_df = merged_df.drop(columns=[drop_column])
            return merged_df
        except Exception as e:
            logging.error("Error in merge_dataframes method: %s", str(e))
            raise

    @staticmethod
    def dropna(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drops the rows where any of the specified columns in the DataFrame have missing values.

        Parameters:
        df (pd.DataFrame): The DataFrame to drop rows from.
        columns (list): The columns to check for missing values.

        Returns:
        pd.DataFrame: The DataFrame with the rows dropped.
        """
        try:
            return df.dropna(subset=columns)
        except Exception as e:
            logging.error("Error in dropna method: %s", str(e))
            raise

class GeoDataProcessor:
    """
    A class used to process geographical data in a DataFrame.

    ...

    Attributes
    ----------
    geolocator : Nominatim
        a geolocator object from the geopy library
    geocode : RateLimiter
        a rate limiter for the geolocator

    Methods
    -------
    get_location(row):
        Returns the location of a given row based on its latitude and longitude.
    get_address_component(loc, component):
        Returns a specific component of an address.
    process_dataframe(df):
        Adds location, city, district, and province columns to a DataFrame.
    """

    def __init__(self, user_agent="poi_search", min_delay_seconds=0.05):
        self.geolocator = Nominatim(user_agent=user_agent)
        self.geocode = RateLimiter(self.geolocator.reverse, min_delay_seconds=min_delay_seconds)

    def get_location(self, row):
        """Returns the location of a given row based on its latitude and longitude."""
        try:
            return self.geocode((row['Latitude'], row['Longitude']))
        except Exception as e:
            logging.error("Error in get_location method: %s", str(e))
            raise

    def get_address_component(self, loc, component):
        """Returns a specific component of an address."""
        try:
            keys = ['city', 'town', 'residential', 'hamlet', 'suburb', 'village', 'amenity']
            if component in keys:
                for key in keys:
                    component_value = loc.raw.get('address', {}).get(key)
                    if component_value is not None:
                        return component_value
            else:
                return loc.raw.get('address', {}).get(component, None)
        except Exception as e:
            logging.error("Error in get_address_component method: %s", str(e))
            raise

    def process_dataframe(self, df):
        """Adds location, city, district, and province columns to a DataFrame."""
        try:
            df['location'] = df.apply(self.get_location, axis=1)
            df['city'] = df['location'].apply(lambda loc: self.get_address_component(loc, 'city'))
            df['district'] = df['location'].apply(lambda loc: self.get_address_component(loc, 'state_district'))
            df['province'] = df['location'].apply(lambda loc: self.get_address_component(loc, 'state'))
            return df
        except Exception as e:
            logging.error("Error in process_dataframe method: %s", str(e))
            raise

class DataFrameProcessor:
    @staticmethod
    def group_and_calculate(df, groupby_column, agg_dict, new_column, divisor):
        """Groups the DataFrame by a column and calculates the sum of another column."""
        grouped_df = df.groupby(groupby_column).agg(agg_dict)
        grouped_df.reset_index(inplace=True)
        grouped_df[new_column] = grouped_df['FinalValue'] / divisor
        return grouped_df

    @staticmethod
    def find_repeating_values(df, subset_columns):
        """Finds rows with repeating values in specified columns."""
        repeating_values = df[df.duplicated(subset=subset_columns[0], keep=False) & ~df.duplicated(subset=subset_columns[1], keep=False)]
        return repeating_values

    @staticmethod
    def print_dataframe(df, columns):
        """Prints specified columns of the DataFrame."""
        print(df[columns].reset_index(drop=True))

    @staticmethod
    def warn_incorrect_location_data(df, location_columns, customer_column):
        """
        Finds customers with incorrect location data and issues a warning.

        Parameters:
        df (pd.DataFrame): The DataFrame to check for incorrect location data.
        location_columns (list): The columns to check for missing values.
        customer_column (str): The column that identifies the customer.

        Returns:
        None
        """
        try:
            null_locations = df[df[location_columns].isnull().any(axis=1)]
            null_location_customers = ', '.join(null_locations[customer_column].unique())
            if null_location_customers:
                print(f"Customers enter their Location data incorrectly {null_location_customers}")
                warnings.warn(f"Customers enter their Location data incorrectly {null_location_customers}")
            return null_locations
        except Exception as e:
            logging.error("Error in warn_incorrect_location_data method: %s", str(e))
            raise
    @staticmethod
    def remove_incorrect_location_data(df, grouped_df, distributor_column, location_columns, customer_column):
        """
        Finds customers with incorrect location data and removes them from the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to remove customers from.
        grouped_df (pd.DataFrame): The DataFrame to find customers in.
        distributor_column (str): The distributor column in the grouped DataFrame.
        location_columns (list): The location columns to check for missing values.
        customer_column (str): The customer column in the DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with the customers removed.
        """
        null_locations = df[df[location_columns].isnull().any(axis=1)]
        null_location_customers = null_locations[customer_column].unique()
        matched_rows = grouped_df[grouped_df[distributor_column].isin(null_location_customers)]
        if matched_rows.empty:
            print("Dealers with no location data or incorrect location data has no any sales in this month.")
        else:
            print("Dealers with no location data or incorrect location data. But he/they have sales in this month.")
            print(matched_rows)
        df = df[~df[customer_column].isin(null_location_customers)]
        print("The row has been removed from the dfGeoCustomer dataframe.")
        return df

    @staticmethod
    def group_and_manage(df, groupby_column, agg_column, new_column, operation='count'):
        """Groups the DataFrame by a column and performs an operation on another column."""
        if operation == 'count':
            grouped_df = df.groupby(groupby_column)[agg_column].count()
        elif operation == 'list':
            grouped_df = df.groupby(groupby_column)[agg_column].apply(list)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        grouped_df = grouped_df.sort_values(ascending=False)
        grouped_df.rename(new_column, inplace=True)
        return grouped_df

class DataFrameAnalyzer:
    @staticmethod
    def print_unique_counts(df, columns):
        for column in columns:
            print(f"Unique {column} count is {len(df[column].unique())}")

    @staticmethod
    def print_total_sales(df, month):
        total_sales = df[df['Date'].dt.month == month]['FinalValue'].sum()
        print(f"The total sales in month {month} is {total_sales}.")
        print(f"The total sales in month {month} is {(total_sales/ 1e6).round(2)} million.")

    @staticmethod
    def print_top_contributors(df, groupby_column, sort_column, n=20):
        top_contributors = df.groupby(groupby_column)[sort_column].sum().sort_values(ascending=False).head(n)
        print(top_contributors)

    @staticmethod
    def print_info(df):
        print(df.info())

    @staticmethod
    def print_fraction_of_sales_no_location(df, month, location_columns=['Latitude', 'Longitude'], distributor_column='DistributorCode'):
        
        """
        Calculates and prints the sales contribution of distributors with no location data.

        Parameters:
        df (pd.DataFrame): The DataFrame to calculate sales from.
        distributor_column (str): The name of the distributor column in the DataFrame.
        null_rowsUser (list): The list of distributor codes with no location data.

        Returns:
        None
        """
        null_rowsUser = df.loc[df[location_columns].isnull().all(axis=1), distributor_column].unique()
        print('Dealers with no location data ')
        print(null_rowsUser)
        total_sales = df[df['Date'].dt.month == month]['FinalValue'].sum()
        total_sales_no_location = df[df[distributor_column].isin(null_rowsUser)]['FinalValue'].sum()
        # print(f'They Contribute {total_sales_no_location} ({((total_sales_no_location*1e-6).round(2))} million) amount of sales')
        print(f'They Contribute {total_sales_no_location} ({((total_sales_no_location*1e-6).round(2))} million) amount of sales')
        fraction = ((total_sales_no_location / total_sales)*100).round(3)
        print(f"The fraction of sales from Dealers with no location data over total sales in month {month} is {fraction} %.")
