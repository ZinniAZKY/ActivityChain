import pandas as pd
import glob
from datetime import datetime, timedelta, date
from datetime import time

# 合并PT数据
column_names = ['Person id', 'trip id', 'subtrip id', 'date', 'lon', 'lat', 'gender', 'age', 'address code', 'occupation', 'purpose', 'mag factor1', 'mag factor2', 'transport mode']
all_data = pd.DataFrame()
csv_files = glob.glob('/Users/zhangkunyi/Downloads/ShizuokaPT原始数据(未融合)/**/*.csv', recursive=True)

for file in csv_files:
    data = pd.read_csv(file, header=None)
    data.columns = column_names
    grouped = data.groupby(['Person id', 'trip id', 'subtrip id'])
    processed_data = pd.concat([grouped.first(), grouped.last()])
    processed_data.reset_index(inplace=True)
    all_data = pd.concat([all_data, processed_data])
    print("finished", file)

all_data.to_csv('/Users/zhangkunyi/Downloads/ShizuokaMergedPT.csv', index=False)


# 新增一列activity并根据purpose修改值
data = pd.read_csv('/Users/zhangkunyi/Downloads/ShizuokaMergedPT.csv')
data['purpose'] = data['purpose'].astype(str)
data['activity'] = data['purpose']

activity_codes = pd.read_csv('/Users/zhangkunyi/Downloads/ShizuokaPurposeCode.csv', dtype={'code': str})
activity_dict = pd.Series(activity_codes.activity.values, index=activity_codes.code).to_dict()
staypoint_codes = pd.read_csv('/Users/zhangkunyi/Downloads/PurposeToActivity.csv')
activity_dict_next = pd.Series(staypoint_codes.activity.values,
                               index=staypoint_codes.purpose).to_dict()
data['activity'] = data['activity'].replace(activity_dict)
data.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)


def process_activities(group):
    # 检验Home停留点是否为2
    count_home = 0
    for i, row in group.iterrows():
        if row['activity'] == 'Home':
            count_home += 1
        elif count_home != 0:
            if count_home != 2:
                print(
                    f"Person id {group.iloc[0]['Person id']} has a sequence of {count_home} consecutive 'Home' activities.")
            count_home = 0
    if count_home != 0 and count_home != 2:
        print(f"Person id {group.iloc[0]['Person id']} has a sequence of {count_home} consecutive 'Home' activities.")

    # 更新停留点
    for activity in activity_dict.values():
        if activity != 'Home':
            non_home_followed_by_different = (group['activity'] == activity) & \
                                             (group['activity'].shift(-1) != activity) & \
                                             (group['activity'].shift(-1) != 'Home')

            if non_home_followed_by_different.any():
                print(
                    f"Person id {group.iloc[0]['Person id']} has non-Home activity '{activity}' followed by a different activity.")

            mask = (group['activity'] == activity) & (group['activity'].shift(-1) == 'Home')
            following_homes = mask.shift(1).fillna(False)

            while following_homes.sum() > 0:
                group.loc[following_homes, 'activity'] = activity_dict_next[activity]
                following_homes = (group['activity'] == 'Home') & following_homes.shift(1).fillna(False)

    return group


data = data.groupby('Person id').apply(process_activities).reset_index(drop=True)
data.to_csv('/Users/zhangkunyi/Downloads/ShizuokaActivityPT.csv', index=False)





# 生成 trip chain
data = pd.read_csv('/Users/zhangkunyi/Downloads/ShizuokaActivityPT.csv')
data['date'] = pd.to_datetime(data['date'])
data.sort_values(['Person id', 'trip id', 'subtrip id'], inplace=True)
data_copy = data.copy()
data_copy['end_date'] = data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['date'].shift(-1)
data_copy['end_activity'] = data_copy.groupby(['Person id', 'trip id', 'subtrip id'])['activity'].shift(-1)
data_copy = data_copy.dropna(subset=['end_date', 'end_activity'])

# 转换为15分钟时间链
data_copy['date'] = pd.to_datetime(data_copy['date']).dt.time
data_copy['end_date'] = pd.to_datetime(data_copy['end_date']).dt.time
intervals = pd.date_range(start='00:00', end='23:59', freq='15min').time
result = pd.DataFrame(index=data_copy['Person id'].unique(), columns=intervals)

for person_id in data_copy['Person id'].unique():
    person_data = data_copy[data_copy['Person id'] == person_id]

    for idx, row in person_data.iterrows():
        activity_duration = {}
        print("Starting one person.")

        for interval in intervals:
            interval_start = interval
            interval_end = (datetime.combine(date.today(), interval) + timedelta(minutes=15)).time() if interval != \
                                                                                                        intervals[
                                                                                                            -1] else time(
                23, 59, 59)

            if max(row['date'], interval_start) < min(row['end_date'], interval_end):
                overlap_start = max(row['date'], interval_start)
                overlap_end = min(row['end_date'], interval_end)
                overlap_duration = datetime.combine(date.today(), overlap_end) - datetime.combine(date.today(),
                                                                                                  overlap_start)

                if interval not in activity_duration:
                    activity_duration[interval] = []
                activity_duration[interval].append((overlap_duration, row['activity']))

        for interval, durations in activity_duration.items():
            max_duration = max(durations, key=lambda x: x[0])
            result.loc[person_id, interval] = max_duration[1]

result.reset_index(inplace=True)
result.rename(columns={'index': 'Person id'}, inplace=True)
result.to_csv('/Users/zhangkunyi/Downloads/ShizuokaTransformedPT.csv', index=False)
