import pandas as pd
import glob
from datetime import datetime, timedelta, date
from datetime import time
import multiprocessing as mp
from multiprocessing import Pool
import os


# # 转换原始PT为OD数据（只保留一天，其他天数删除）
# def process_df(file):
#     column_names = ['Person id', 'trip id', 'subtrip id', 'date', 'lon', 'lat', 'gender', 'age', 'address code', 'occupation', 'purpose', 'mag factor1', 'mag factor2', 'transport mode']
#     input_df = pd.read_csv(file, header=None)
#     input_df.columns = column_names
#
#     input_df['date'] = pd.to_datetime(input_df['date'])
#     input_df.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)
#     input_df.reset_index(drop=True, inplace=True)
#
#     start_date = input_df.loc[0, 'date'].date()
#     if input_df.loc[0, 'date'].time() != pd.Timestamp(0).time():
#         print(f"Start time for file {file} is not at the beginning of a day.")
#
#     original_length = len(input_df)
#     input_df = input_df[input_df['date'].dt.date == start_date]
#     if len(input_df) < original_length:
#         print(f"Some rows removed for file {file}. Original length: {original_length}. New length: {len(input_df)}.")
#
#     grouped_df = input_df.groupby(['Person id', 'trip id', 'subtrip id'])
#     processed_df = pd.concat([grouped_df.first(), grouped_df.last()])
#     processed_df.reset_index(inplace=True)
#
#     return processed_df
#
#
# def main():
#     base_path = '/Users/zhangkunyi/Downloads/PTFolder'
#     directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and 'PTOriginal' in d]
#
#     for directory in directories:
#         input_csv = glob.glob(f'{base_path}/{directory}/**/*.csv', recursive=True)
#
#         with mp.Pool(processes=6) as pool:
#             result_df = pool.map(process_df, input_csv)
#
#         all_df = pd.concat(result_df)
#         all_df.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)
#         groups = all_df.groupby('Person id')
#
#         for name, group in groups:
#             group = group.sort_values('date').reset_index(drop=True)
#             group = group.iloc[1:-1]
#             count_df = group['date'].value_counts().reset_index()
#             count_df.columns = ['date', 'count']
#             if any(count_df['count'] != 2):
#                 print(f"Person {name} does not have all date occurs twice")
#
#         all_df.to_csv(f'{base_path}/{directory.replace("PTOriginal", "PTMerged")}.csv', index=False)
#
#
# if __name__ == '__main__':
#     main()


NUM_WORKERS = 4


# 新增一列activity并根据purpose修改值
def process_activities(group, activity_dict, activity_dict_next):
    # 更新所有非House停留点之前检验连续停留点是否为2。如果发生移动但目的为不明，则可能出现连续的2的奇数倍个停留点但实际为无法判别的移动。如果停留点为奇数则可能为错误。
    count_house = 0
    for i, row in group.iterrows():
        if row['activity'] == 'House':
            count_house += 1
        elif count_house != 0:
            if count_house % 2 != 0:
                print(
                    f"Person id {group.iloc[0]['Person id']} has odd number of {count_house} consecutive 'House' activities.")
            count_house = 0
    if count_house != 0 and count_house % 2 != 0:
        print(
            f"Person id {group.iloc[0]['Person id']} has odd number of {count_house} consecutive 'House' as the last activities.")

    # 检测非停留点的两个移动是否连续存在并更新停留点
    for activity in set(activity_dict.values()):
        if activity != 'House':
            non_house_followed_by_different = (group['activity'] == activity) & \
                                              (group['activity'].shift(-1) != activity) & \
                                              (group['activity'].shift(-1) != 'House')

            if non_house_followed_by_different.any():
                print(
                    f"Person id {group.iloc[0]['Person id']} has non-House activity '{activity}' followed by a different activity.")

            mask = (group['activity'] == activity) & (group['activity'].shift(-1) == 'House')
            following_houses = mask.shift(1).fillna(False)

            while following_houses.sum() > 0:
                group.loc[following_houses, 'activity'] = activity_dict_next[activity]
                following_houses = (group['activity'] == 'House') & following_houses.shift(1).fillna(False)

    return group


def process_file(file):
    data = pd.read_csv(file)
    data['purpose'] = data['purpose'].astype(str)
    data['activity'] = data['purpose']
    data.sort_values(['Person id', 'trip id', 'subtrip id', 'date'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    staypoint_codes = pd.read_csv('/Users/zhangkunyi/Downloads/RegionPurposeCode/PurposeToActivity.csv')
    activity_dict_next = pd.Series(staypoint_codes.activity.values, index=staypoint_codes.purpose).to_dict()
    base = os.path.basename(file)
    base = base.replace("PTMerged", "")
    activity_codes_file = '/Users/zhangkunyi/Downloads/RegionPurposeCode/' + base
    activity_codes = pd.read_csv(activity_codes_file, dtype={'code': str})
    activity_dict = pd.Series(activity_codes.purpose.values, index=activity_codes.code).to_dict()

    data['activity'] = data['activity'].replace(activity_dict)
    data = data.groupby('Person id').apply(process_activities, activity_dict, activity_dict_next).reset_index(drop=True)

    base, ext = os.path.splitext(file)
    output_file = base + "ActivityPT" + ext
    data.to_csv(output_file, index=False)


def main():
    path = '/Users/zhangkunyi/Downloads/PTFolder/PTMerged'
    files = glob.glob(os.path.join(path, '*.csv'))

    with Pool(NUM_WORKERS) as p:
        p.map(process_file, files)


if __name__ == "__main__":
    main()





# # Define the path for the input CSV files
# path = '/Users/zhangkunyi/Downloads/PTFolder/PTMerged'
#
# # Get a list of all CSV files in the specified directory
# files = glob.glob(os.path.join(path, '*.csv'))
#
# # Process all files
# for file in files:
#     process_file(file)




# # 生成 trip chain
# data = pd.read_csv('/Users/zhangkunyi/Downloads/ShizuokaActivityPT.csv')
# data['date'] = pd.to_datetime(data['date'])
# data.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)
# data_copy = data.copy()
# data_copy['end_date'] = data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['date'].shift(-1)
# data_copy['end_activity'] = data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['activity'].shift(-1)
# data_copy = data_copy.dropna(subset=['end_date', 'end_activity'])
#
# # 转换为15分钟时间链
# data_copy['date'] = pd.to_datetime(data_copy['date']).dt.time
# data_copy['end_date'] = pd.to_datetime(data_copy['end_date']).dt.time
# intervals = pd.date_range(start='00:00', end='23:59', freq='15min').time
# result = pd.DataFrame(index=data_copy['Person id'].unique(), columns=intervals)
#
# for person_id in data_copy['Person id'].unique():
#     person_data = data_copy[data_copy['Person id'] == person_id]
#     activity_duration = {interval: [] for interval in intervals}
#
#     for idx, row in person_data.iterrows():
#         for interval in intervals:
#             interval_start = interval
#             interval_end = (datetime.combine(date.today(), interval) + timedelta(minutes=15)).time() if interval != \
#                                                                                                         intervals[
#                                                                                                             -1] else time(
#                 23, 59, 59)
#
#             if max(row['date'], interval_start) < min(row['end_date'], interval_end):
#                 overlap_start = max(row['date'], interval_start)
#                 overlap_end = min(row['end_date'], interval_end)
#                 overlap_duration = datetime.combine(date.today(), overlap_end) - datetime.combine(date.today(),
#                                                                                                   overlap_start)
#
#                 activity_duration[interval].append((overlap_duration, row['activity']))
#
#     for interval, durations in activity_duration.items():
#         if durations:
#             max_duration_activity = max(durations, key=lambda x: x[0])[1]
#             result.loc[person_id, interval] = max_duration_activity
#             print("finished")
#
#
# result.reset_index(inplace=True)
# result.rename(columns={'index': 'Person id'}, inplace=True)
# result.to_csv('/Users/zhangkunyi/Downloads/ShizuokaTransformedPT.csv', index=False)
