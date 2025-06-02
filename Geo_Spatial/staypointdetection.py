# Find the optimum epsilon value for DBSCAN

import math

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from datetime import datetime
import numpy as np
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance
    distance = R * c *1000
    return distance
# n1 =1
# n2 = 20

# Test the function with actual coordinates
#lat1, lon1 = latitude_list[n1], longitude_list[n1]
lat1, lon1 = 6.890117, 79.86391
lat2, lon2 = 6.890026, 79.86392

print(haversine(lat1, lon1, lat2, lon2), "m")

## DBSCAN Clustering Geo Locations ::

class StayPointDetector:
    KMS_PER_RADIAN = 6371.0088  # Earth's radius in kilometers

    def __init__(self, df):
        """Initialize the processor with a dataframe."""
        self.df = df

    def perform_dbscan(self, epsilon=0.005, min_samples=1, algorithm='ball_tree', metric='haversine'):
        coords = np.radians(self.df[['Latitude', 'Longitude']].values)
        epsilon = epsilon / self.KMS_PER_RADIAN

        db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm=algorithm, metric=metric)
        db.fit(coords)

        self.df['Cluster'] = db.labels_

        cluster_means = self.df.groupby('Cluster')[['Latitude', 'Longitude']].transform('mean')
        self.df['Constant Latitude'] = cluster_means['Latitude']
        self.df['Constant Longitude'] = cluster_means['Longitude']
        # self.df = self.df.join(cluster_means, on='Cluster', rsuffix=' Mean')
        
    def calculate_distance(self, row):
        if pd.notnull(row['Next Latitude']) and pd.notnull(row['Next Longitude']):
            return geodesic((row['Constant Latitude'], row['Constant Longitude']), 
                            (row['Next Latitude'], row['Next Longitude'])).meters
        else:
            return np.nan

    def process_dbscan(self, distance_threshold=25,Group_time_difference_threshold=2, time_difference_threshold=5 ):
        self.df['RecievedDate'] = pd.to_datetime(self.df['RecievedDate'])
        self.df['Date'] = self.df['RecievedDate'].dt.date
        self.df['Location Change'] = (self.df['Constant Latitude'].diff().ne(0) | self.df['Constant Longitude'].diff().ne(0)).cumsum()
        grouped = self.df.groupby(['UserCode', 'Date', 'Location Change'])
        
        # grouped = self.df.groupby(['UserCode', 'Date', 'Constant Latitude', 'Constant Longitude'])
        self.df['First Time'] = grouped['RecievedDate'].transform('first')
        self.df['Last Time'] = grouped['RecievedDate'].transform('last')
        self.df['Spending Time (minutes)'] = (self.df['Last Time'] - self.df['First Time']).dt.total_seconds() / 60

        mask = self.df['Spending Time (minutes)'] >= Group_time_difference_threshold
        self.df = self.df[mask]

        self.df = self.df.drop_duplicates(subset=['UserCode', 'Date', 'Constant Latitude', 'Constant Longitude'])

        self.df['Next Latitude'] = self.df['Constant Latitude'].shift(-1)
        self.df['Next Longitude'] = self.df['Constant Longitude'].shift(-1)
        self.df['Distance'] = self.df.apply(self.calculate_distance, axis=1)
        self.df['Distance'] = self.df['Distance'].shift(1)

        self.df['Next Spending Time (minutes)'] = self.df['Spending Time (minutes)'].shift(-1)
        mask = (self.df['Distance'] <= distance_threshold) & (self.df['Next Spending Time (minutes)'] >= time_difference_threshold)
        
        self.df.loc[mask, 'Mean Latitude'] = self.df.loc[mask, ['Constant Latitude', 'Next Latitude']].mean(axis=1)
        self.df.loc[mask, 'Mean Longitude'] = self.df.loc[mask, ['Constant Longitude', 'Next Longitude']].mean(axis=1)
        # self.df.loc[mask, 'Time Difference (minutes)'] = self.df.loc[mask, ['Spending Time (minutes)']].sum(axis=1)
        self.df.loc[mask, 'Time Difference (minutes)'] = self.df.loc[mask, ['Spending Time (minutes)', 'Next Spending Time (minutes)']].sum(axis=1)
        
        # Fill NaN values in 'Mean Latitude' and 'Mean Longitude' with 'Constant Latitude' and 'Constant Longitude'
        self.df['Mean Latitude'] = self.df['Mean Latitude'].fillna(self.df['Constant Latitude'])
        self.df['Mean Longitude'] = self.df['Mean Longitude'].fillna(self.df['Constant Longitude'])
        
        # When the condition is not met, set 'Time Difference (minutes)' to be the 'Spending Time (minutes)' of the current row
        self.df.loc[~mask, 'Time Difference (minutes)'] = self.df.loc[~mask, 'Spending Time (minutes)']
        self.df = self.df.reset_index(drop=True)

    def process(self, epsilon=0.005, distance_threshold=25, Group_time_difference_threshold=2, time_difference_threshold=5):
        self.perform_dbscan(epsilon=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
        self.process_dbscan(distance_threshold=distance_threshold, time_difference_threshold=time_difference_threshold)

    def test_perform_dbscan(self):
        self.perform_dbscan()
        assert 'Cluster' in self.df.columns
        assert 'Constant Latitude' in self.df.columns
        assert 'Constant Longitude' in self.df.columns
class GeoSpatialAnalyzer:
    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0088  

    def __init__(self, df):
        """Initialize the processor with a dataframe."""
        self.df = df.copy()

    @staticmethod
    def calculate_distances(df1, df2):
        """
        Calculate haversine distances between two dataframes.
        
        Parameters:
        df1, df2: DataFrames containing 'Latitude' and 'Longitude' columns.
        
        Returns:
        distances: A 2D numpy array containing haversine distances.
        """
        customer_coords = np.radians(df1[['Latitude', 'Longitude']].values)
        cluster_coords = np.radians(df2[['Mean Latitude', 'Mean Longitude']].drop_duplicates().values)

        distances = haversine_distances(customer_coords, cluster_coords) * GeoSpatialAnalyzer.EARTH_RADIUS_KM

        return distances

    @staticmethod
    def assign_closest_usercode(df1, df2, distances):
        """
        Assign the closest user code to each row in df1 based on the distances.
        
        Parameters:
        df1, df2: DataFrames containing 'Latitude' and 'Longitude' columns.
        distances: A 2D numpy array containing haversine distances between df1 and df2.
        
        Returns:
        df1: DataFrame with an additional column 'Closet UserCode'.
        closest_clusters_index: Indices of the closest user codes.
        unique_clusters: DataFrame of unique clusters.
        """
        closest_clusters_index = np.argmin(distances, axis=1)
        unique_clusters = df2[['Mean Latitude', 'Mean Longitude', 'UserCode']].drop_duplicates().reset_index(drop=True)

        df1['Closet UserCode'] = unique_clusters.loc[closest_clusters_index, 'UserCode'].values

        return df1, closest_clusters_index, unique_clusters

    @staticmethod
    def assign_visited_status(df, distances, threshold=0.1):
        """
        Assign a visited status to each row in df based on the distances.
        
        Parameters:
        df: DataFrame containing 'Latitude' and 'Longitude' columns.
        distances: A 2D numpy array containing haversine distances.
        threshold: A threshold for the distance to consider a location as visited.
        
        Returns:
        df: DataFrame with an additional column 'Visited'.
        """
        min_distances = np.min(distances, axis=1)
        df['Visited'] = min_distances <= threshold

        return df

    @staticmethod
    def count_visited_status(df):
        """
        Count the visited status in df.
        
        Parameters:
        df: DataFrame containing a 'Visited' column.
        
        Returns:
        visited_counts: A Series containing counts of visited status.
        """
        visited_counts = df['Visited'].value_counts()

        return visited_counts

    @staticmethod
    def count_false_per_user(df):
        """
        Count the number of 'False' in the 'Visited' column for each user.
        
        Parameters:
        df: DataFrame containing 'Visited' and 'Closet UserCode' columns.
        
        Returns:
        user_false_counts: A Series containing counts of 'False' for each user.
        """
        user_false_counts = df.groupby('Closet UserCode')['Visited'].apply(lambda x: (x == False).sum())

        return user_false_counts

    @staticmethod
    def assign_visited_usercode(df1, df2, distances, threshold=0.1):
        """
        Assign a visited user code to each row in df1 based on the distances.
        
        Parameters:
        df1, df2: DataFrames containing 'Latitude' and 'Longitude' columns.
        distances: A 2D numpy array containing haversine distances.
        threshold: A threshold for the distance to consider a location as visited.
        
        Returns:
        df1: DataFrame with an additional column 'Visited UserCode'.
        """
        closest_clusters_index = np.argmin(distances, axis=1)
        unique_clusters = df2[['Mean Latitude', 'Mean Longitude', 'UserCode']].drop_duplicates().reset_index(drop=True)

        min_distances = np.min(distances, axis=1)
        df1['Visited UserCode'] = np.where(min_distances <= threshold, unique_clusters.loc[closest_clusters_index, 'UserCode'].values, np.nan)

        return df1

    @staticmethod
    def get_DistributorNo_for_unvisited_ClosetUserCode(df):
        """
        Get the distributor number for unvisited closet user code.
        
        Parameters:
        df: DataFrame containing 'Visited' and 'Closet UserCode' columns.
        
        Returns:
        no_values: A Series containing distributor numbers for unvisited closet user code.
        """
        unvisited_df = df[df['Visited'] == False]
        no_values = unvisited_df.groupby('Closet UserCode')['No'].apply(lambda x: list(set(x)))
        return no_values

    @staticmethod
    def check_DistributorNo_in_other_ClosetUsercode(no_values):
        """
        Check if there are common distributor numbers in other closet user codes.
        
        Parameters:
        no_values: A Series containing distributor numbers for unvisited closet user code.
        """
        common_found = False
        for usercode, no_list in no_values.items():
            other_no_values = no_values.drop(usercode).values
            other_no_list = [item for sublist in other_no_values for item in sublist]
            common_no_values = set(no_list).intersection(other_no_list)
            if len(common_no_values) != 0:
                common_found = True
                print(f'Common "No" values in "{usercode}" and other UserCodes: {common_no_values}')
                print(f'Count of common "No" values: {len(common_no_values)}')
        if not common_found:
            print("No Common Distributors in Nearest BPOs'")
class GeoSpatialEvaluator:
    def __init__(self, df, coords_cols=['Latitude', 'Longitude'], cluster_col='Cluster', visited_col='Visited'):
        self.df = df
        self.coords_cols = coords_cols
        self.cluster_col = cluster_col
        self.visited_col = visited_col

    def evaluate_performance(self, true_labels):
        """
        Evaluate the performance of the geospatial analysis.
        """
        # Predicted labels
        pred_labels = self.df[self.visited_col]

        # Precision, Recall, F1 Score
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        print('Confusion Matrix:')
        print(cm)

        # Average Distance to Nearest Cluster for 'Visited' Locations
        visited_df = self.df[self.df[self.visited_col] == True]
        avg_distance = visited_df['Distance'].mean()
        print(f'Average Distance to Nearest Cluster for "Visited" Locations: {avg_distance}')

    @staticmethod
    def check_location_within_clusters(dfGeo, dfGPS, epsilon=0.05):
        """
        Check if each location in dfGeoCustomer is within a certain distance (defined by epsilon) from any cluster in dfSFA_GPSDataDBSCAN1.
        """
        # Convert Latitude and Longitude into radians for haversine_distances
        customer_coords_eval = np.radians(dfGeo[['Latitude', 'Longitude']].values)
        cluster_coords_eval = np.radians(dfGPS[['Latitude', 'Longitude']].drop_duplicates().values)

        # Calculate haversine_distances and convert to kilometers
        kms_per_radian = 6371.0088 # Earth's radius in kms
        distances = haversine_distances(customer_coords_eval, cluster_coords_eval) * kms_per_radian

        #dfGeoCustomerCopy = dfGeoCustomer.copy()
        # Check if the minimum distance is within the error tolerance (epsilon)
        dfGeo['Visited'] = np.min(distances, axis=1) <= epsilon

        dfGeo_True = dfGeo[dfGeo['Visited'] == True]

        return dfGeo_True

    def calculate_silhouette_score(self):
        labels = self.df[self.cluster_col]
        coords = np.radians(self.df[self.coords_cols].values)
        silhouette = silhouette_score(coords, labels, metric='euclidean')
        return silhouette

    def calculate_davies_bouldin_index(self):
        labels = self.df[self.cluster_col]
        coords = np.radians(self.df[self.coords_cols].values)
        db_score = davies_bouldin_score(coords, labels)
        return db_score
    

