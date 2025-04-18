import pandas as pd
import numpy as np
from functools import reduce
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import datetime
import dateutil
from typing import List
from pandas.tseries.frequencies import to_offset
from typing import Literal

def initialize_xslx_data(initial_df):
    df = initial_df.copy()
    df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index, 'coerce', dayfirst=True)
    df = df[~df.index.isna()]
    
    return df

    title = df.iloc[3]['Unnamed: 0']
    df = df.drop([2, 3])
    data_groups = {}
    current_group = df.iloc[0].values[0]

    for index, value in zip(df.iloc[0].index, df.iloc[0].values):
        if (pd.isna(value)):
            data_groups[current_group].append(index)
        else:
            current_group = value
            data_groups[current_group] = [index]

    data_dfs = []
    index = pd.to_datetime(df['Unnamed: 0'][2:-3], dayfirst=True)

    for group, cols in zip(list(data_groups.keys())[1:], list(data_groups.values())[1:]):
        data_df = df[cols].copy()
        data_df.columns = data_df.iloc[1].values
        data_df = data_df.drop([0, 1])[:-3]
        data_df = data_df.set_index(index)
        data_df.index = pd.to_datetime(data_df.index)
        data_df.index.names = ['timestamp']
        data_dfs.append((group, data_df))

    return title, data_groups, data_dfs

def get_nan_intervals(series: pd.Series) -> List[pd.Series]:
    na_mask = series.isna()
    groups = (na_mask != na_mask.shift()).cumsum()
    nan_intervals = series[na_mask].groupby(groups[na_mask])

    return [interval for _, interval in nan_intervals]

def predict_xslx(data: pd.DataFrame) -> pd.Series:
    df_empty = data.copy()
    df_empty.loc[:, :] = np.nan
    for col_name in data.columns:
        col = data[col_name]
        nan_intervals = get_nan_intervals(col)
        if (len(nan_intervals) == 0):
            continue
        for interval in nan_intervals:
            if (len(interval)) == 0:
                continue
            # конец заполненных данных
            avaiable_train_end_index = col.index.get_indexer([interval.index[0]])
            # доступные для обучения
            avaiable_train = col.iloc[:avaiable_train_end_index[0]]
            delta = pd.Timedelta(days=avaiable_train.index[-1].weekday())
            # timestamp начала недели
            week_start = (avaiable_train.index[-1] - delta).normalize()
            # индекс начала недели
            train_start = avaiable_train.index.get_indexer([week_start], method='backfill')
            # трейн данные с начала недели
            train = avaiable_train[train_start[0]:]
            seasonal_periods = 24
            min_len = seasonal_periods * 2
            if len(train) < min_len:
                # добавляются данные с прошлой недели если не хватате для обучения
                if (len(avaiable_train[:train.index[0]][:-1]) < min_len - len(train)):
                    continue
                else:
                    train = pd.concat([avaiable_train[:train.index[0]][:-1][:len(train) - (min_len+1):-1],train]).sort_index()
            seasonal_periods = 24
            eps = 10e-03
            train = np.log(train.fillna(train.mean()).map(lambda x: max(x, 1)) + eps)
            fit = ExponentialSmoothing(
                train.values,
                trend=None,
                seasonal='mul',
                seasonal_periods=seasonal_periods,
                damped_trend=False,
            ).fit()
            left_i = col.index.get_loc(interval.index[0])
            right_i = col.index.get_loc(interval.index[-1])
            # предикт данных включая граничные значения
            predict_index = col.iloc[left_i-1:right_i+2].index
            # запись
            df_empty.loc[predict_index,col_name] = np.exp(fit.forecast(len(interval) + 2))
            # df.loc[predict_index,col_name] = np.exp(int(fit.forecast(len(interval) + 2)))

    return df_empty


def initialize_csv_data(path: str, year: int):
    # puid_df = pd.read_csv(path)
    # puid_df['keys'] = puid_df['keys'].apply(ast.literal_eval)
    # puid_df['values'] = puid_df['values'].apply(ast.literal_eval)
    # puid_df['meta'] = puid_df['meta'].apply(ast.literal_eval)
    # meta = list(map(json.loads, puid_df['meta'][0]))
    # titles = list(map(
    #     lambda x: f'{"Прямое" if  x["direction"] == 1 else "Обратное"}, полоса {x["lane"]}', meta))
    # keys = puid_df['keys'][0][0]
    # general_key_indices = [keys.index('volume'), keys.index(
    #     'speed'), keys.index('occupancy')]

    df = pd.read_csv(path)
    df = df.set_index('index')
    df.index = pd.to_datetime(df.index)

    year_range = pd.date_range(
        f'{year}-01-01 00:00', f'{year}-12-31 23:55', freq='5Min')
    df = df.reindex(year_range)

    return df


def get_na_intervals(df: pd.DataFrame) -> pd.DataFrame:
    df['is_nan'] = df['volume'].isna().astype(int)
    df['group'] = (df['is_nan'].diff() == 1).cumsum()
    df = df.reset_index()
    nan_intervals = df[df['is_nan'] == 1].groupby(
        'group').agg(start=('index', 'first'), end=('index', 'last'))
    df = df.set_index('index')
    df = df.drop(['is_nan', 'group'], axis=1)
    nan_intervals['duration'] = nan_intervals['end'] - nan_intervals['start']
    return nan_intervals


def fill_na(df: pd.DataFrame, nan_interval: (pd.Timestamp, pd.Timestamp, pd.Timedelta), method: Literal['exp_smooth', 'moving_average']) -> pd.DataFrame:
    start, end, duration = nan_interval.start, nan_interval.end, nan_interval.duration
    if (method == 'exp_smooth'):
        # avaiable train data
        max_train_len = len(df[~df['volume'].isna()][:start])
        if max_train_len == 0:
            return
        # check type of interval
        if (duration > pd.Timedelta(days=5)):
            return
        elif (duration > pd.Timedelta(hours=10)):
            train = df[~df['volume'].isna()][start -
                                             pd.Timedelta(days=7):start][:-1]
            first_timestamp = train.index[-1]
            train_start = (first_timestamp -
                           pd.Timedelta(days=first_timestamp.weekday())).normalize()
            train = train[train_start:]
            nan_counts = (int)((end - start) / pd.Timedelta(minutes=5))
            fit = ExponentialSmoothing(
                train['volume'],
                trend=None,
                seasonal='add',
                seasonal_periods=24 * 60 / 5,
                damped_trend=False,
            ).fit()
            forecast = pd.DataFrame(pd.Series(fit.forecast(
                steps=nan_counts)), columns=['forecast'])
            return forecast
        else:
            pass


def process_csv_dataframe(path: str):
    df = initialize_csv_data(path, 2024)
    nan_intervals = get_na_intervals(df)
    forecast_df = pd.DataFrame(index=df.index)

    for row in nan_intervals.itertuples():
        forecast = fill_na(df, row, method='exp_smooth')
        if (forecast):
            forecast_df['volume'] = forecast


def process_xslx_dataframe(path):
    initial_df = pd.read_excel(path, header=[2,3])
    df = initialize_xslx_data(initial_df)
    predicted = predict_xslx(df)
    initial_df = initial_df.set_index(initial_df.columns[0])
    initial_df.index = pd.to_datetime(initial_df.index, 'coerce', dayfirst=True)
    initial_df = initial_df[~initial_df.index.isna()]
    initial_df

    return initial_df, predicted
