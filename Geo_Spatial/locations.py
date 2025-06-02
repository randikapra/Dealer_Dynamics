# Phase 2.0
import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd

# Data Preparation
# Assuming dfSFA_OrdersJanCopy is your DataFrame
dfSFA_OrdersJanCopy['TotalFinalValue'] = dfSFA_OrdersJanCopy.groupby('DistributorCode')['FinalValue'].transform('sum')
dfGeoCustomer_Full = dfSFA_OrdersJanCopy.drop_duplicates(subset='DistributorCode')

# Function to create a popup string
def create_popup(row, detailed=False):
    """
    Create a popup string or HTML iframe for folium map markers.

    Parameters:
    row (pd.Series): A row from DataFrame containing marker data.
    detailed (bool): If True, create detailed HTML popup, else simple string.

    Returns:
    folium.Popup or str: Popup object for detailed info, or string for simple popup.
    """
    if detailed:
        html = f"""
        <h2>Dealer Information</h2>
        <p><strong>Dealer Number:</strong> {row['DistributorCode']}</p>
        <p><strong>Location:</strong> </p> 
            <p>Latitude: {row['Latitude']}, Longitude: {row['Longitude']}</p>
        <p><strong>BPO:</strong> </p> 
            <p>BPO Code: {row['UserCode']}</p>
        <p><strong>Total Sales in Month:</strong></p>
            <p>Sales: {row['TotalFinalValue']} ({row['TotalFinalValue'] * 1e-6:.2f} Million)</p>
        <p><strong>First Sale Date:</strong> </p> 
            <p>Date: {row['Date']}</p>
        <!-- Add more dealer-specific information here -->
        <p><a href='http://www.dealerwebsite.com' target='_blank'>Visit Dealer Website</a></p>
        """
        iframe = folium.IFrame(html=html, width=200, height=100)
        return folium.Popup(iframe)
    else:
        return f"Dealer: {row['No']}, Latitude: {row['Latitude']}, Longitude: {row['Longitude']}"

# Initialize the map
def initialize_map(df, location=[7.8731, 80.7718], zoom_start=13):
    """
    Initialize a folium map centered on the mean location of the DataFrame or a specified location.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Latitude' and 'Longitude' columns.
    location (list): Default location to center the map if DataFrame is missing coordinates.
    zoom_start (int): Initial zoom level for the map.

    Returns:
    folium.Map: Initialized map object.
    """
    try:
        map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    except KeyError:
        map_center = location
    return folium.Map(location=map_center, zoom_start=zoom_start)

# Add markers to the map
def add_markers_to_map(df, map_object, tooltip=False):
    """
    Add markers to the folium map with optional tooltips.

    Parameters:
    df (pd.DataFrame): DataFrame containing marker data.
    map_object (folium.Map): Map object to add markers to.
    tooltip (bool): If True, use tooltips instead of popups.

    Returns:
    folium.Map: Map object with added markers.
    """
    marker_cluster = MarkerCluster().add_to(map_object)
    for idx, row in df.iterrows():
        try:
            if tooltip:
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']],
                    radius=5,
                    color="red",
                    fill=True,
                    tooltip=create_popup(row)
                ).add_to(marker_cluster)
            else:
                popup = create_popup(row, detailed=True)
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']],
                    radius=5,
                    color="red",
                    fill=True,
                    popup=popup
                ).add_to(marker_cluster)
        except KeyError as e:
            print(f"Missing data for marker: {e}")

# Main function to create the map
def create_customer_map(df, tooltip=False, filename='customer_map.html'):
    """
    Create a customer map with markers and save it as an HTML file.

    Parameters:
    df (pd.DataFrame): DataFrame containing customer data.
    tooltip (bool): If True, use tooltips instead of popups.
    filename (str): Filename to save the HTML map.

    Returns:
    folium.Map: Customer map with markers.
    """
    map_object = initialize_map(df)
    add_markers_to_map(df, map_object, tooltip)
    # map_object.save(filename)
    return map_object

# Create a HeatMap
def create_heat_map(df, map_object, filename='heat_map.html'):
    """
    Create a HeatMap and add it to the provided folium map object.

    Parameters:
    df (pd.DataFrame): DataFrame containing the latitude and longitude columns.
    map_object (folium.Map): Folium map object to which the HeatMap will be added.
    filename (str): Filename to save the HTML map.

    Returns:
    folium.Map: The map object with the HeatMap added.
    """
    try:
        heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]
        HeatMap(heat_data).add_to(map_object)
        # map_object.save(filename)
    except KeyError as e:
        print(f"DataFrame is missing required column: {e}")

# Add the Choropleth layer using the aggregated data
def add_choropleth_layer(df, map_object, geojson_url, filename='choropleth_map.html'):
    """
    Add a Choropleth layer to the folium map using aggregated data.

    Parameters:
    df (pd.DataFrame): DataFrame containing district data.
    map_object (folium.Map): Map object to add the Choropleth layer to.
    geojson_url (str): URL to the GeoJSON file for the Choropleth layer.
    filename (str): Filename to save the HTML map.

    Returns:
    folium.Map: Map object with the Choropleth layer added.
    """
    try:
        df['district'] = df['district'].str.replace(' District', '')  # Clean district names
        dfNumericNo = df.groupby('district')['No'].count().reset_index(name='NumericNo')
        folium.Choropleth(
            geo_data=geojson_url,
            name='choropleth',
            data=dfNumericNo,
            columns=['district', 'NumericNo'],
            key_on='feature.properties.electoralDistrict',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Numeric Value Representation'
        ).add_to(map_object)
        # map_object.save(filename)
    except KeyError as e:
        print(f"DataFrame is missing required column: {e}")
# Assuming dfGeoCustomer is already defined and loaded with data
map_optimized = create_customer_map(dfGeoCustomer, tooltip=True, filename='optimized_customer_map.html')

# Create maps
map_heat = initialize_map(dfGeoCustomer)
create_heat_map(dfGeoCustomer, map_heat)

geojson_url = "https://raw.githubusercontent.com/thejeshgn/srilanka/master/electoral_districts_map/LKA_electrol_districts.geojson"
map_choropleth = initialize_map(dfGeoCustomer, zoom_start=8)
add_choropleth_layer(dfGeoCustomer, map_choropleth, geojson_url)

map_detailed_markers = create_customer_map(dfGeoCustomer_Full, tooltip=False, filename='detailed_markers_map.html')


dfSFA_OrdersJanCopy = DataPreprocessor.merge_dataframes(dfSFA_OrdersJanCopy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')
dfSFA_OrdersFebCopy = DataPreprocessor.merge_dataframes(dfSFA_OrdersFebCopy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')
dfSFA_OrdersMarCopy = DataPreprocessor.merge_dataframes(dfSFA_OrdersMarCopy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')
dfSFA_Orders1Copy = DataPreprocessor.merge_dataframes(dfSFA_Orders1Copy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')
dfSFA_OrdersAllCopy = DataPreprocessor.merge_dataframes(dfSFA_OrdersAllCopy, dfGeoCustomer, left_on='DistributorCode', right_on='No', how='left', drop_column='No')

dfSFA_OrdersJanCopy = DataPreprocessor.dropna(dfSFA_OrdersJanCopy, ['Latitude', 'Longitude'])
dfSFA_OrdersFebCopy = DataPreprocessor.dropna(dfSFA_OrdersFebCopy, ['Latitude', 'Longitude'])
dfSFA_OrdersMarCopy = DataPreprocessor.dropna(dfSFA_OrdersMarCopy, ['Latitude', 'Longitude'])
dfSFA_Orders1Copy = DataPreprocessor.dropna(dfSFA_Orders1Copy, ['Latitude', 'Longitude'])
dfSFA_OrdersAllCopy = DataPreprocessor.dropna(dfSFA_OrdersAllCopy, ['Latitude', 'Longitude'])
DataFrameprocessor = DataFrameProcessor()
grouped_df_FinalSale = DataFrameprocessor.group_and_calculate(dfSFA_OrdersJanCopy, 'DistributorCode', {'FinalValue': 'sum', 'Latitude': 'first', 'Longitude': 'first'}, 'FinalValue_Millions (MM)', 1e6)
print('Customers with sales in January month ::')
DataFrameprocessor.print_dataframe(grouped_df_FinalSale, ['DistributorCode','FinalValue_Millions (MM)'])
print('Same total sales in a month with different Dealers')
repeating_finalvalue_unique_latitude = DataFrameprocessor.find_repeating_values(grouped_df_FinalSale, ['FinalValue', 'Latitude'])
DataFrameprocessor.print_dataframe(repeating_finalvalue_unique_latitude, ['DistributorCode', 'FinalValue', 'FinalValue_Millions (MM)'])



# Phase 1.0
from branca.element import Template, MacroElement

# Define the template for the selector
template = """
{% macro html(this, kwargs) %}
<div style="position: fixed; top: 50px; left: 50px; z-index: 9999; padding: 5px; background: white; border-radius: 5px;">
    <select id="layer-select">
        <option value="map_heat">Heat Map</option>
        <option value="map_choropleth">Choropleth Map</option>
        <option value="map_detailed_markers">Detailed Markers Map</option>
    </select>
</div>
<script>
    document.getElementById('layer-select').addEventListener('change', function(e) {
        let selectedLayer = e.target.value;
        if (selectedLayer === 'map_heat') {
            map_heat_layer.addTo(map);
            map.removeLayer(map_choropleth_layer);
            map.removeLayer(map_detailed_markers_layer);
        } else if (selectedLayer === 'map_choropleth') {
            map_choropleth_layer.addTo(map);
            map.removeLayer(map_heat_layer);
            map.removeLayer(map_detailed_markers_layer);
        } else if (selectedLayer === 'map_detailed_markers') {
            map_detailed_markers_layer.addTo(map);
            map.removeLayer(map_heat_layer);
            map.removeLayer(map_choropleth_layer);
        }
    });
</script>
{% endmacro %}
"""

geojson_url = "https://raw.githubusercontent.com/thejeshgn/srilanka/master/electoral_districts_map/LKA_electrol_districts.geojson"
# Add the selector to the map
selector = MacroElement()
selector._template = Template(template)

# Create the map object
map = initialize_map(dfGeoCustomer)

# Create the layers
map_heat = create_heat_map(dfGeoCustomer, map)
map_choropleth = add_choropleth_layer(dfGeoCustomer, map, geojson_url)
map_detailed_markers = add_markers_to_map(dfGeoCustomer, map, tooltip=True)

# Add the selector to the map
map.get_root().add_child(selector)

# Save the map to an HTML file
# map.save('map_with_selector.html')



# map_optimized.get_root().add_child(selector)

# # Assuming dfGeoCustomer is already defined and loaded with data
# map_optimized = create_customer_map(dfGeoCustomer, tooltip=True, filename='optimized_customer_map.html')

# # Create maps
# map_heat = initialize_map(dfGeoCustomer)
# create_heat_map(dfGeoCustomer, map_heat)

# map_choropleth = initialize_map(dfGeoCustomer, zoom_start=8)
# add_choropleth_layer(dfGeoCustomer, map_choropleth, geojson_url)

# map_detailed_markers = create_customer_map(dfGeoCustomer_Full, tooltip=False, filename='detailed_markers_map.html')



from branca.element import Template, MacroElement

# Define the template for the selector
template = """
{% macro html(this, kwargs) %}
<div style="position: fixed; top: 50px; left: 50px; z-index: 9999; padding: 5px; background: white; border-radius: 5px;">
    <select id="layer-select">
        <option value="map_heat">Heat Map</option>
        <option value="map_choropleth">Choropleth Map</option>
        <option value="map_detailed_markers">Detailed Markers Map</option>
    </select>
</div>
<script>
    document.getElementById('layer-select').addEventListener('change', function(e) {
        let selectedLayer = e.target.value;
        if (selectedLayer === 'map_heat') {
            map_heat_layer.addTo(map);
            map.removeLayer(map_choropleth_layer);
            map.removeLayer(map_detailed_markers_layer);
        } else if (selectedLayer === 'map_choropleth') {
            map_choropleth_layer.addTo(map);
            map.removeLayer(map_heat_layer);
            map.removeLayer(map_detailed_markers_layer);
        } else if (selectedLayer === 'map_detailed_markers') {
            map_detailed_markers_layer.addTo(map);
            map.removeLayer(map_heat_layer);
            map.removeLayer(map_choropleth_layer);
        }
    });
</script>
{% endmacro %}
"""

geojson_url = "https://raw.githubusercontent.com/thejeshgn/srilanka/master/electoral_districts_map/LKA_electrol_districts.geojson"
# Add the selector to the map
selector = MacroElement()
selector._template = Template(template)

# Create the map object
map = initialize_map(dfGeoCustomer)

# Create the layers
map_heat = create_heat_map(dfGeoCustomer, map)
# map_choropleth = add_choropleth_layer(dfGeoCustomer, map, geojson_url)
map_detailed_markers = add_markers_to_map(dfGeoCustomer, map, tooltip=True)

# Add the selector to the map
map.get_root().add_child(selector)

# Save the map to an HTML file
# map.save('map_with_selector.html')



# map_optimized.get_root().add_child(selector)

# # Assuming dfGeoCustomer is already defined and loaded with data
# map_optimized = create_customer_map(dfGeoCustomer, tooltip=True, filename='optimized_customer_map.html')

# # Create maps
# map_heat = initialize_map(dfGeoCustomer)
# create_heat_map(dfGeoCustomer, map_heat)

# map_choropleth = initialize_map(dfGeoCustomer, zoom_start=8)
# add_choropleth_layer(dfGeoCustomer, map_choropleth, geojson_url)

# map_detailed_markers = create_customer_map(dfGeoCustomer_Full, tooltip=False, filename='detailed_markers_map.html')


# def create_popup(row):
#     """
#     Create a popup string or HTML iframe for folium map markers.

#     Parameters:
#     row (pd.Series): A row from DataFrame containing marker data.

#     Returns:
#     folium.Popup: Popup object with appropriate detail level.
#     """
#     # Start with the dealer information header
#     html = "<h2>Dealer Information</h2>"

#     # Add dealer number if present
#     if 'DistributorCode' in row:
#         html += f"<p><strong>Dealer Number:</strong> {row['DistributorCode']}</p>"

#     # Add location coordinates
#     html += f"<p><strong>Location:</strong></p><p>Latitude: {row['Latitude']}, Longitude: {row['Longitude']}</p>"

#     # Add BPO code if present
#     if 'UserCode' in row:
#         html += f"<p><strong>BPO:</strong></p><p>BPO Code: {row['UserCode']}</p>"

#     # Add total sales if present
#     if 'TotalFinalValue' in row:
#         html += f"<p><strong>Total Sales in Month:</strong></p><p>Sales: {row['TotalFinalValue']} ({row['TotalFinalValue'] * 1e-6:.2f} Million)</p>"

#     # Add first sale date if present
#     if 'Date' in row:
#         html += f"<p><strong>First Sale Date:</strong></p><p>Date: {row['Date']}</p>"

#     # Add dealer website link
#     html += "<p><a href='http://www.dealerwebsite.com' target='_blank'>Visit Dealer Website</a></p>"

#     # Create the iframe with the constructed HTML
#     iframe = folium.IFrame(html=html, width=200, height=100)
#     return folium.Popup(iframe)

def create_popup(row, map_type='full'):
    if map_type == 'full':
        html = f"""
        <h2>Dealer Information</h2>
        <p><strong>Dealer Number:</strong> {row['DistributorCode']}</p>
        <p><strong>Location:</strong> </p> 
            <p>Latitude: {row['Latitude']}, Longitude: {row['Longitude']}</p>
        <p><strong>BPO:</strong> </p> 
            <p>BPO Code: {row['UserCode']}</p>
        <p><strong>Total Sales in Month:</strong></p>
            <p>Sales: {row['TotalFinalValue']} ({row['TotalFinalValue'] * 1e-6:.2f} Million)</p>
        <p><strong>First Sale Date:</strong> </p> 
            <p>Date: {row['Date']}</p>
        <!-- Add more dealer-specific information here -->
        <p><a href='http://www.dealerwebsite.com' target='_blank'>Visit Dealer Website</a></p>
        """
    elif map_type == 'partial':
        html = f"""
        <h2>Dealer Information</h2>
        <p><strong>Location:</strong> </p> 
            <p>Latitude: {row['Latitude']}, Longitude: {row['Longitude']}</p>
        <!-- Add more dealer-specific information here -->
        <p><a href='http://www.dealerwebsite.com' target='_blank'>Visit Dealer Website</a></p>
        """
    else:
        # Default simple popup
        return f"Dealer: {row['No']}, Latitude: {row['Latitude']}, Longitude: {row['Longitude']}"

    iframe = folium.IFrame(html=html, width=200, height=100)
    return folium.Popup(iframe)

def initialize_map(df, location=[7.8731, 80.7718], zoom_start=13):
    try:
        map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    except KeyError:
        map_center = location
    return folium.Map(location=map_center, zoom_start=zoom_start)

# Add markers to the map
def add_markers_to_map(df, map_object, tooltip=False):
    marker_cluster = MarkerCluster().add_to(map_object)
    for idx, row in df.iterrows():
        try:
            if tooltip:
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']],
                    radius=5,
                    color="red",
                    fill=True,
                    tooltip=create_popup(row)
                ).add_to(marker_cluster)
            else:
                popup = create_popup(row, map_type = 'partial')
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']],
                    radius=5,
                    color="red",
                    fill=True,
                    popup=popup
                ).add_to(marker_cluster)
        except KeyError as e:
            print(f"Missing data for marker: {e}")

# Main function to create the map
def create_customer_map(df, tooltip=False, filename='customer_map.html'):
    map_object = initialize_map(df)
    add_markers_to_map(df, map_object, tooltip)
    # map_object.save(filename)
    return map_object

# Create a HeatMap
def create_heat_map(df, map_object, filename='heat_map.html'):
    try:
        heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]
        HeatMap(heat_data).add_to(map_object)
        # map_object.save(filename)
    except KeyError as e:
        print(f"DataFrame is missing required column: {e}")

def add_choropleth_layer(df, map_object, geojson_url, filename='choropleth_map.html'):
    try:
        df['district'] = df['district'].str.replace(' District', '')  # Clean district names
        dfNumericNo = df.groupby('district')['No'].count().reset_index(name='NumericNo')
        folium.Choropleth(
            geo_data=geojson_url,
            name='choropleth',
            data=dfNumericNo,
            columns=['district', 'NumericNo'],
            key_on='feature.properties.electoralDistrict',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Numeric Value Representation'
        ).add_to(map_object)
        # map_object.save(filename)
    except KeyError as e:
        print(f"DataFrame is missing required column: {e}")
 
def generate_maps_for_province(df, province_name):
    # Filter the DataFrame for the selected province
    df_province = df[df['province'] == province_name]

    # Check if the filtered DataFrame is empty or if it contains NaN values
    if df_province.empty or df_province[['Latitude', 'Longitude']].isnull().values.any():
        raise ValueError(f"No valid data found for the province: {province_name}")

    # Initialize the map centered around the province's coordinates
    province_location = [df_province['Latitude'].mean(), df_province['Longitude'].mean()]
    map_object = initialize_map(df_province, location=province_location, zoom_start=9)

    # Create a marker map for the province
    map_markers = create_customer_map(df_province)
    # map_markers.save(f'{province_name}_markers_map.html')

    # Create a heat map for the province
    map_heat = initialize_map(df_province, location=province_location, zoom_start=9)
    create_heat_map(df_province, map_heat)
    # map_heat.save(f'{province_name}_heat_map.html')

    # Return the map object for further use if needed
    return map_object, map_markers, map_heat, 

# Example usage:
province_maps, province_maps_marker, province_maps_heat, = generate_maps_for_province(dfGeoCustomer, 'Western Province')


def generate_maps_for_district(df, district_name):
    # Filter the DataFrame for the selected district
    df_district = df[df['district'] == district_name]

    # Check if the filtered DataFrame is empty or if it contains NaN values
    if df_district.empty or df_district[['Latitude', 'Longitude']].isnull().values.any():
        raise ValueError(f"No valid data found for the district: {district_name}")

    # Initialize the map centered around the district's coordinates
    district_location = [df_district['Latitude'].mean(), df_district['Longitude'].mean()]
    map_object = initialize_map(df_district, location=district_location, zoom_start=10)

    # Create a marker map for the district
    map_markers = create_customer_map(df_district)
    # map_markers.save(f'{district_name}_markers_map.html')

    # Create a heat map for the district
    map_heat = initialize_map(df_district, location=district_location, zoom_start=10)
    create_heat_map(df_district, map_heat)
    # map_heat.save(f'{district_name}_heat_map.html')

    # Return the map objects for further use if needed
    return map_object, map_markers, map_heat

# Example usage:
district_maps, district_maps_marker, district_maps_heat = generate_maps_for_district(dfGeoCustomer, 'Colombo')


