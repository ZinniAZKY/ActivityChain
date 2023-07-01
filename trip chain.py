import pandas as pd
import glob
from datetime import datetime, timedelta, date, time
import time
from datetime import time



# # 合并PT数据
# # Define column names
# column_names = ['Person id', 'trip id', 'subtrip id', 'date', 'lon', 'lat', 'gender', 'age', 'address code', 'occupation', 'purpose', 'mag factor1', 'mag factor2', 'transport mode']
#
# # Create an empty DataFrame for collecting the processed data
# all_data = pd.DataFrame()
#
# # List all csv files in the current directory or specify your own path
# csv_files = glob.glob('/Users/zhangkunyi/Downloads/ShizuokaPT原始数据(未融合)/**/*.csv', recursive=True)
#
# for file in csv_files:
#     # Read csv file without header
#     data = pd.read_csv(file, header=None)
#
#     # Assign the column names
#     data.columns = column_names
#
#     # Group the DataFrame by "Person id", "trip id" and "subtrip id"
#     # Then get the first and last row of each group
#     grouped = data.groupby(['Person id', 'trip id', 'subtrip id'])
#     processed_data = pd.concat([grouped.first(), grouped.last()])
#
#     # Reset the index
#     processed_data.reset_index(inplace=True)
#
#     # Append processed_data to all_data
#     all_data = pd.concat([all_data, processed_data])
#     print("finished", file)
#
# # Save the DataFrame to a new csv file, without the index
# all_data.to_csv('/Users/zhangkunyi/Downloads/ShizuokaMergedPT.csv', index=False)




# # 新增一列activity并根据purpose修改值
# # Load your data
# data = pd.read_csv('/Users/zhangkunyi/Downloads/ShizuokaMergedPT.csv')
#
# # Convert 'purpose' column to string
# data['purpose'] = data['purpose'].astype(str)
#
# # Initialize a new column for 'activity'
# data['activity'] = data['purpose']
#
# # Replace purpose codes with corresponding activity labels
# activity_dict = {'99': 'Home', '1': 'Commute', '2': 'Go School', '3': 'Back Home',
#                  '4': 'Go Shopping_D', '5': 'Go Shopping_ND', '6': 'Go Entertainment',
#                  '7': 'Transfer Person', '8': 'Go Hospital', '9': 'Go Tourism',
#                  '10': 'Go For Private', '11': 'Trip To Deliver', '12': 'Go For Meeting',
#                  '13': 'Go For Repair', '14': 'Go For Agriculture', '15': 'Go For Others'}
# data['activity'] = data['activity'].replace(activity_dict)
#
# # Order the DataFrame by 'Person id'
# data.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)
#
# # Determine the next activity
# activity_dict_next = {'Commute': 'Office', 'Go School': 'School', 'Back Home': 'Home',
#                       'Go Shopping_D': 'Daily Shopping', 'Go Shopping_ND': 'Nondaily Shopping',
#                       'Go Entertainment': 'Eating', 'Transfer Person': 'Welcome',
#                       'Go Hospital': 'Hospital', 'Go Tourism': 'Tourism', 'Go For Private': 'Private',
#                       'Trip To Deliver': 'Deliver', 'Go For Meeting': 'Meeting',
#                       'Go For Repair': 'Fixing', 'Go For Agriculture': 'Agriculture', 'Go For Others': 'Others'}
#
#
# def update_activity(group):
#     for activity in activity_dict.values():
#         if activity != 'Home':
#             # Only mark the row as end of a series if the next activity is 'Home'
#             mask = (group['activity'] == activity) & (group['activity'].shift(-1) == 'Home')
#             following_homes = mask.shift(1).fillna(False)
#             while following_homes.sum() > 0:
#                 group.loc[following_homes, 'activity'] = activity_dict_next[activity]
#                 following_homes = (group['activity'] == 'Home') & following_homes.shift(1).fillna(False)
#                 print('Finished')
#     return group
#
#
# # Apply the function to each group
# data = data.groupby('Person id').apply(update_activity).reset_index(drop=True)
#
# # Save the DataFrame
# data.to_csv('/Users/zhangkunyi/Downloads/ShizuokaActivityPT.csv', index=False)








# 生成 trip chain
data = pd.read_csv('/Users/zhangkunyi/Downloads/ShizuokaActivityPT.csv')

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Order the DataFrame by 'Person id', 'trip id', and 'subtrip id'
data.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)

# Create a copy of the data DataFrame to avoid modifying it directly
data_copy = data.copy()

# Shift the 'date' and 'activity' columns up by one row within each group
data_copy['end_date'] = data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['date'].shift(-1)
data_copy['end_activity'] = data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['activity'].shift(-1)

# Filter out the rows where 'end_date' or 'end_activity' is NaN (these will be the original even rows)
data_copy = data_copy.dropna(subset=['end_date', 'end_activity'])

# 转换为15分钟时间链
# Convert 'date' and 'end_date' to datetime and extract the time part
data_copy['date'] = pd.to_datetime(data_copy['date']).dt.time
data_copy['end_date'] = pd.to_datetime(data_copy['end_date']).dt.time

# Generate the 15-minutes time intervals
intervals = pd.date_range(start='00:00', end='23:59', freq='15min').time

# Initialize a new DataFrame for the results
result = pd.DataFrame(index=data_copy['Person id'].unique(), columns=intervals)

# Loop through each person
for person_id in data_copy['Person id'].unique():
    person_data = data_copy[data_copy['Person id'] == person_id]

    # Loop through each activity
    for idx, row in person_data.iterrows():
        activity_duration = {}
        print("Starting one person.")

        # Loop through each 15-minutes interval
        for interval in intervals:
            interval_start = interval
            interval_end = (datetime.combine(date.today(), interval) + timedelta(minutes=15)).time() if interval != \
                                                                                                        intervals[
                                                                                                            -1] else time(
                23, 59, 59)

            # Check if this activity overlaps with the interval
            if max(row['date'], interval_start) < min(row['end_date'], interval_end):
                overlap_start = max(row['date'], interval_start)
                overlap_end = min(row['end_date'], interval_end)
                overlap_duration = datetime.combine(date.today(), overlap_end) - datetime.combine(date.today(),
                                                                                                  overlap_start)

                # Save the overlap duration
                if interval not in activity_duration:
                    activity_duration[interval] = []
                activity_duration[interval].append((overlap_duration, row['activity']))

        # Update the results DataFrame
        for interval, durations in activity_duration.items():
            # Find the activity with maximum duration
            max_duration = max(durations, key=lambda x: x[0])
            result.loc[person_id, interval] = max_duration[1]

result.reset_index(inplace=True)
result.rename(columns={'index': 'Person id'}, inplace=True)
result.to_csv('/Users/zhangkunyi/Downloads/ShizuokaTransformedPT.csv', index=False)







