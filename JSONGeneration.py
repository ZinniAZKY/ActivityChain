import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
from datetime import datetime, timedelta, date, time
from multiprocessing import Pool
import numpy as np
import glob
import json

# # Spatial Join
# df = pd.read_csv('/Users/zhangkunyi/Downloads/PTFolder/PTActivity/Tokyo2008PTActivity.csv')
# gdf_points = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['lon'], df['lat'])])
# gdf_points.crs = "EPSG:4326"
# gdf_admin = gpd.read_file('/Users/zhangkunyi/Downloads/H30_SmallZone_gis/H30_szone.shp')
# gdf_admin = gdf_admin.to_crs("EPSG:4326")
# gdf_admin_szone = gdf_admin[['szone', 'geometry']]
# gdf_joined = gpd.sjoin(gdf_points, gdf_admin_szone, how="left", op='intersects')
# gdf_joined.to_csv('/Users/zhangkunyi/Downloads/PTFolder/Spatial_Join_Tokyo.csv', index=False)


# # 生成small zone
# def generate_trip_chain(file):
#     act_data = pd.read_csv(file)
#     act_data['szone'] = act_data['szone'].replace('', '99999').fillna('99999')
#     act_data['date'] = pd.to_datetime(act_data['date'])
#     act_data.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)
#     act_data_copy = act_data.copy()
#     act_data_copy['end_date'] = act_data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['date'].shift(-1)
#     act_data_copy['end_szone'] = act_data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['szone'].shift(-1)
#     act_data_copy = act_data_copy.dropna(subset=['end_date', 'end_szone'])
#
#     # Convert to 15-minute time chain
#     act_data_copy['date'] = pd.to_datetime(act_data_copy['date']).dt.time
#     act_data_copy['end_date'] = pd.to_datetime(act_data_copy['end_date']).dt.time
#     intervals = pd.date_range(start='00:00', end='23:59', freq='15min').time
#     chain_result = pd.DataFrame(index=act_data_copy['Person id'].unique(), columns=intervals)
#
#     for person_id in act_data_copy['Person id'].unique():
#         person_data = act_data_copy[act_data_copy['Person id'] == person_id]
#         act_duration = {interval: [] for interval in intervals}
#
#         for idx, row in person_data.iterrows():
#             for interval in intervals:
#                 act_start = interval
#                 act_end = (datetime.combine(date.today(), interval) + timedelta(minutes=15)).time() if interval != \
#                                                                                                        intervals[
#                                                                                                            -1] else time(
#                     23, 59, 59)
#
#                 if max(row['date'], act_start) < min(row['end_date'], act_end):
#                     act_start_update = max(row['date'], act_start)
#                     act_end_update = min(row['end_date'], act_end)
#                     act_duration_update = datetime.combine(date.today(), act_end_update) - datetime.combine(
#                         date.today(),
#                         act_start_update)
#
#                     act_duration[interval].append((act_duration_update, row['szone']))
#
#         for interval, durations in act_duration.items():
#             if durations:
#                 act_duration_max = max(durations, key=lambda x: x[0])[1]
#                 chain_result.loc[person_id, interval] = act_duration_max
#             print("finished")
#
#     chain_result.reset_index(inplace=True)
#     chain_result.rename(columns={'index': 'Person id'}, inplace=True)
#
#     base, ext = os.path.splitext(file)
#     chain_dir = "/Users/zhangkunyi/Downloads"
#     os.makedirs(chain_dir, exist_ok=True)
#     chain_file = chain_dir + "/" + os.path.basename(base).replace('Spatial_Join', 'SZone_Trip_Chain') + ext
#
#     chain_result.to_csv(chain_file, index=False)
#
#
# def process_files_in_parallel(file_paths):
#     num_pool = 6
#     with Pool(num_pool) as p:
#         p.map(generate_trip_chain, file_paths)
#
#
# def split_file(file_path, num_splits):
#     # Read the original file
#     df = pd.read_csv(file_path)
#
#     # Get unique person IDs and the counts of records for each
#     unique_person_ids = df['Person id'].unique()
#     person_ids_per_split = np.array_split(unique_person_ids, num_splits)
#     files = []
#
#     # Split the DataFrame based on person ID and write each split to a new file
#     for i, person_ids in enumerate(person_ids_per_split):
#         split_df = df[df['Person id'].isin(person_ids)]
#         split_file_path = f"{file_path.split('.csv')[0]}_part_{i}.csv"
#         split_df.to_csv(split_file_path, index=False)
#         files.append(split_file_path)
#
#     return files
#
#
# if __name__ == "__main__":
#     # Example file paths list, replace with your actual paths
#     original_file = '/Users/zhangkunyi/Downloads/PTFolder/Spatial_Join_Tokyo.csv'
#     num_splits = 30  # Number of files to split into
#     split_files = split_file(original_file, num_splits)
#     process_files_in_parallel(split_files)


# 合并生成small zone
# def combine_output_files(output_dir, combined_file_name):
#     # Create a pattern to match all output files
#     pattern = os.path.join(output_dir, "*_part_*.csv")  # Adjust pattern as needed
#     file_list = glob.glob(pattern)
#
#     # Initialize an empty list to store DataFrames
#     dfs = []
#
#     # Loop through the list of file paths & append each file to the DataFrame list
#     for file in file_list:
#         df = pd.read_csv(file)
#         dfs.append(df)
#
#     # Concatenate all DataFrames in the list
#     combined_df = pd.concat(dfs, ignore_index=True)
#
#     # Save the combined DataFrame to a new CSV file
#     combined_file_path = os.path.join(output_dir, combined_file_name)
#     combined_df.to_csv(combined_file_path, index=False)
#     print(f"Combined file saved to {combined_file_path}")
#
#
# # Assuming the output files are stored in '/Users/zhangkunyi/Downloads'
# output_directory = '/Users/zhangkunyi/Downloads'
# combined_file_name = 'Combined_SZone_Trip_Chain.csv'
#
# # Call the function to combine files
# combine_output_files(output_directory, combined_file_name)


# Step 1: Read the CSV files
df1 = pd.read_csv('/Users/zhangkunyi/Downloads/PTFolder/PTChain/Tokyo2008PTChain.csv')
df2 = pd.read_csv('/Users/zhangkunyi/Downloads/Combined_SZone_Trip_Chain.csv')
df3 = pd.read_csv('/Users/zhangkunyi/Downloads/PTFolder/PTActivity/Tokyo2008PTActivity.csv')
df3 = df3.drop_duplicates(subset='Person id')

# Mapping dictionaries
gender_map = {1: 'Male', 2: 'Female'}
occu_maps = {1: 'Agriculture_Worker', 10: 'Other_Occupation', 11: 'Student', 12: 'Student', 13: 'Student',
             14: 'Housewife', 15: 'Unemployed', 16: 'Other_Occupation', 2: 'Labor_Worker',
             3: 'Sales_Worker', 4: 'Service_Worker', 5: 'Traffic_Worker', 6: 'Security_Worker',
             7: 'Office_Worker', 8: 'Technical_Worker', 9: 'Managerial_Worker', 99: 'Unclear'}
df3['gender'] = df3['gender'].map(gender_map)
df3['occupation'] = df3['occupation'].map(occu_maps)
df3['age'] = df3['age'] * 5

# Merge df1 (activity) and df2 (small zone) on 'Person id'
merged_df = pd.merge(df1, df2, on='Person id', suffixes=('_activity', '_szone'))

# Merge the combined df with df3 (personal attributes) on 'Person id'
final_df = pd.merge(merged_df, df3, on='Person id')


# Convert the final merged DataFrame to the desired JSON format
def row_to_json(row):
    # Extract personal attributes and prepare the activity and szone series
    json_entry = {
        'person_id': row['Person id'],
        'age': row['age'],
        'gender': row['gender'],
        'occupation': row['occupation'],
        'activity_series': {},
        'szone_series': {}
    }
    # Assuming your interval columns are named consistently, e.g., 'interval1_activity', 'interval1_szone', etc.
    for col in final_df.columns:
        if '_activity' in col:
            interval = col.replace('_activity', '')
            json_entry['activity_series'][interval] = row[col]
        elif '_szone' in col:
            interval = col.replace('_szone', '')
            json_entry['szone_series'][interval] = row[col]
    return json_entry


# Apply the function to each row of the DataFrame and collect the results
json_data = final_df.apply(row_to_json, axis=1).tolist()

# Write the JSON data to a file
with open('/Users/zhangkunyi/Downloads/final_output.json', 'w') as file:
    json.dump(json_data, file, indent=4)
