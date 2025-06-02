## Supply Chain Optimization

import pandas as pd
import pulp

# Calculate the total demand (FinalValue) per province
demand_per_province = df_merged.groupby('province')['FinalValue'].sum().reset_index()
demand_per_province.rename(columns={'FinalValue': 'Demand'}, inplace=True)

# Calculate the mean latitude and longitude for each province
mean_coords_per_province = df_merged.groupby('province')[['Latitude', 'Longitude']].mean().reset_index()

# Merge the demand data with the mean coordinates
warehouse_locations = pd.merge(mean_coords_per_province, demand_per_province, on='province')

# Define the problem
problem = pulp.LpProblem("WarehouseLocationProblem", pulp.LpMinimize)

# Variables: 1 if a warehouse is built in the province, 0 otherwise
warehouse_vars = pulp.LpVariable.dicts("Warehouse", warehouse_locations['province'], 0, 1, pulp.LpBinary)

# Objective function: Minimize the sum of the distances to each province weighted by demand
# You will need to define a way to calculate the distance between provinces
# For example, using the Haversine formula or another appropriate distance measure
problem += pulp.lpSum([warehouse_vars[province] * warehouse_locations.loc[warehouse_locations['province'] == province, 'Demand'].values[0] for province in warehouse_locations['province']])

# Constraints: Define your constraints here, such as minimum number of warehouses or regional constraints

# Solve the problem
problem.solve()

# Output the results
for v in problem.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)


