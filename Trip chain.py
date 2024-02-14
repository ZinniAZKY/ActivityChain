import pandas as pd
import glob
from datetime import datetime, timedelta, date
from datetime import time
import multiprocessing as mp
from multiprocessing import Pool
import os

# 转换原始PT为OD数据（只保留一天，其他天数删除）
def process_df(file):
    column_names = ['Person id', 'trip id', 'subtrip id', 'date', 'lon', 'lat', 'gender', 'age', 'address code', 'occupation', 'purpose', 'mag factor1', 'mag factor2', 'transport mode']
    input_df = pd.read_csv(file, header=None)
    input_df.columns = column_names

    input_df['date'] = pd.to_datetime(input_df['date'])
    input_df.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)
    input_df.reset_index(drop=True, inplace=True)

    start_date = input_df.loc[0, 'date'].date()
    if input_df.loc[0, 'date'].time() != pd.Timestamp(0).time():
        print(f"Start time for file {file} is not at the beginning of a day.")

    original_length = len(input_df)
    input_df = input_df[input_df['date'].dt.date == start_date]
    if len(input_df) < original_length:
        print(f"Some rows removed for file {file}. Original length: {original_length}. New length: {len(input_df)}.")

    grouped_df = input_df.groupby(['Person id', 'trip id', 'subtrip id'])
    processed_df = pd.concat([grouped_df.first(), grouped_df.last()])
    processed_df.reset_index(inplace=True)

    return processed_df


def main():
    base_path = '/Users/zhangkunyi/Downloads/PTFolder'
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and 'PTOriginal' in d]

    for directory in directories:
        input_csv = glob.glob(f'{base_path}/{directory}/**/*.csv', recursive=True)

        with mp.Pool(processes=4) as pool:
            result_df = pool.map(process_df, input_csv)

        all_df = pd.concat(result_df)
        all_df.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)
        groups = all_df.groupby('Person id')

        for name, group in groups:
            group = group.sort_values('date').reset_index(drop=True)
            group = group.iloc[1:-1]
            count_df = group['date'].value_counts().reset_index()
            count_df.columns = ['date', 'count']
            if any(count_df['count'] != 2):
                print(f"Person {name} does not have all date occurs twice")

        all_df.to_csv(f'{base_path}/{directory.replace("PTOriginal", "PTMerged")}.csv', index=False)


if __name__ == '__main__':
    main()


NUM_POOL = 6


# 生成出行链，原始数据在PTMerged文件夹，生成数据分别在PTActivity和PTChain中，读取的字典在RegionPurposeCode中
# 需要修改staypoint_codes，activity_codes_file，chain_dir，merged_path，act_path路径。
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
            check_next_act = (group['activity'] == activity) & \
                             (group['activity'].shift(-1) != activity) & \
                             (group['activity'].shift(-1) != 'House')

            if check_next_act.any():
                print(
                    f"Person id {group.iloc[0]['Person id']} has non-House activity '{activity}' followed by a different activity.")

            act_mask = (group['activity'] == activity) & (group['activity'].shift(-1) == 'House')
            next_houses = act_mask.shift(1).fillna(False)

            while next_houses.sum() > 0:
                group.loc[next_houses, 'activity'] = activity_dict_next[activity]
                next_houses = (group['activity'] == 'House') & next_houses.shift(1).fillna(False)

    return group


def modify_activity(data):
    # Generate a sequence number for each trip id within each person id
    data['trip_seq'] = data.groupby(['Person id', 'trip id']).cumcount() + 1

    # For each pair of consecutive rows within the same group, check if they have the same activity and transportation mode
    same_activity_transport = (data['activity'] == data['activity'].shift()) & (
            data['transport mode'] == data['transport mode'].shift())

    # Find indices where the next two rows are continuous and have transportation mode 97 and same activity
    indices_to_change = []
    for idx in data.index[same_activity_transport]:
        # Ensure the next two indices are continuous and within the bounds of the DataFrame
        if idx + 2 < len(data) and data.loc[idx + 1, 'Person id'] == data.loc[idx, 'Person id'] and data.loc[idx + 2, 'Person id'] == data.loc[idx, 'Person id']:
            if data.loc[idx + 1, 'transport mode'] == 97 and data.loc[idx + 2, 'transport mode'] == 97:
                if data.loc[idx + 1, 'activity'] == data.loc[idx, 'activity'] and data.loc[idx + 2, 'activity'] == data.loc[idx, 'activity']:
                    indices_to_change.extend([idx + 1, idx + 2])

    # Change 'activity' to '99' for the identified rows
    data.loc[indices_to_change, 'activity'] = '99'
    data.drop(columns=['trip_seq'], inplace=True)

    return data


def reassign_activity(file):
    merged_data = pd.read_csv(file)
    merged_data['purpose'] = merged_data['purpose'].astype(str)
    merged_data['activity'] = merged_data['purpose']
    merged_data.sort_values(['Person id', 'trip id', 'subtrip id', 'date'], inplace=True)
    merged_data.reset_index(drop=True, inplace=True)

    # Preprocess data to modify 'staying at home' activities
    merged_data = modify_activity(merged_data)

    staypoint_codes = pd.read_csv('/Users/zhangkunyi/Downloads/RegionPurposeCode/PurposeToActivity.csv')
    activity_dict_next = pd.Series(staypoint_codes.activity.values, index=staypoint_codes.purpose).to_dict()
    base_act = os.path.basename(file)
    base_act = base_act.replace("PTMerged", "")
    activity_codes_file = '/Users/zhangkunyi/Downloads/RegionPurposeCode/' + base_act
    activity_codes = pd.read_csv(activity_codes_file, dtype={'code': str})
    activity_dict = pd.Series(activity_codes.purpose.values, index=activity_codes.code).to_dict()

    merged_data['activity'] = merged_data['activity'].replace(activity_dict)
    grouped_data = merged_data.groupby('Person id', group_keys=True).apply(process_activities, activity_dict, activity_dict_next).reset_index(drop=True)

    base_act, ext = os.path.splitext(file)
    base_act = base_act.replace("PTMerged", "")
    act_dir = os.path.dirname(base_act) + "/PTActivity"
    os.makedirs(act_dir, exist_ok=True)
    act_file = act_dir + "/" + os.path.basename(base_act) + "PTActivity" + ext
    grouped_data.to_csv(act_file, index=False)


def generate_trip_chain(file):
    # Generate trip chain
    act_data = pd.read_csv(file)
    act_data['date'] = pd.to_datetime(act_data['date'])
    act_data.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)
    act_data_copy = act_data.copy()
    act_data_copy['end_date'] = act_data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['date'].shift(-1)
    act_data_copy['end_activity'] = act_data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['activity'].shift(-1)
    act_data_copy = act_data_copy.dropna(subset=['end_date', 'end_activity'])

    # Convert to 15-minute time chain
    act_data_copy['date'] = pd.to_datetime(act_data_copy['date']).dt.time
    act_data_copy['end_date'] = pd.to_datetime(act_data_copy['end_date']).dt.time
    intervals = pd.date_range(start='00:00', end='23:59', freq='15min').time
    chain_result = pd.DataFrame(index=act_data_copy['Person id'].unique(), columns=intervals)

    for person_id in act_data_copy['Person id'].unique():
        person_data = act_data_copy[act_data_copy['Person id'] == person_id]
        act_duration = {interval: [] for interval in intervals}

        for idx, row in person_data.iterrows():
            for interval in intervals:
                act_start = interval
                act_end = (datetime.combine(date.today(), interval) + timedelta(minutes=15)).time() if interval != \
                                                                                                       intervals[
                                                                                                           -1] else time(
                    23, 59, 59)

                if max(row['date'], act_start) < min(row['end_date'], act_end):
                    act_start_update = max(row['date'], act_start)
                    act_end_update = min(row['end_date'], act_end)
                    act_duration_update = datetime.combine(date.today(), act_end_update) - datetime.combine(
                        date.today(),
                        act_start_update)

                    act_duration[interval].append((act_duration_update, row['activity']))

        for interval, durations in act_duration.items():
            if durations:
                act_duration_max = max(durations, key=lambda x: x[0])[1]
                chain_result.loc[person_id, interval] = act_duration_max
            print("finished")

    chain_result.reset_index(inplace=True)
    chain_result.rename(columns={'index': 'Person id'}, inplace=True)

    base, ext = os.path.splitext(file)
    chain_dir = "/Users/zhangkunyi/Downloads/PTFolder/PTChain"
    os.makedirs(chain_dir, exist_ok=True)
    chain_file = chain_dir + "/" + os.path.basename(base).replace("PTActivity", "PTChain") + ext

    chain_result.to_csv(chain_file, index=False)


def main():
    # 转换purpose为activity
    merged_path = '/Users/zhangkunyi/Downloads/PTFolder/PTMerged'
    merged_files = glob.glob(os.path.join(merged_path, '*.csv'))

    with Pool(NUM_POOL) as p:
        p.map(reassign_activity, merged_files)

    # 生成trip chain
    act_path = '/Users/zhangkunyi/Downloads/PTFolder/PTActivity'
    act_files = glob.glob(os.path.join(act_path, '*.csv'))

    with Pool(NUM_POOL) as p:
        p.map(generate_trip_chain, act_files)


if __name__ == "__main__":
    main()
