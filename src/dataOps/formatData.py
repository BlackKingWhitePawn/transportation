import pandas as pd
import ast
import json
from typing import Union, Literal
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm
import pathlib
import datetime
import numpy as np
from pathlib import Path


def initialize_xslx_data(initial_df):
    df = initial_df.copy()
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


def predict_xslx(df, title):
    predicted_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        data = df[col].to_frame().sort_index()
        data['is_nan'] = data[col].isna().astype(int)
        data['group'] = (data['is_nan'].diff() == 1).cumsum()
        data = data.reset_index()
        nan_intervals = data[data['is_nan'] == 1].groupby(
            'group').agg(start=('timestamp', 'first'), end=('timestamp', 'last'))
        data = data.set_index('timestamp')
        data = data.drop(['is_nan', 'group'], axis=1)
        data = data.astype(float)
        nan_intervals['duration'] = nan_intervals['end'] - \
            nan_intervals['start']
        if (len(nan_intervals) == 0):
            predicted_df[f'{title} - {col} - predict'] = pd.Series(
                    index=predicted_df.index)
            continue
        for interval in tqdm(nan_intervals.itertuples()):
            if (len(interval) == 0):
                predicted_df[f'{col} - predict'] = pd.Series(
                    index=predicted_df.index)
                continue
            start, end, duration = interval.start, interval.end, interval.duration
            train_end = data[:start].index[-2]
            train_start = (
                train_end - pd.Timedelta(days=train_end.weekday())).normalize()
            train_start = data.iloc[data.index.get_indexer(
                [train_start], method='nearest')[0]].name
            train = data[train_start:train_end]
            eps = 10e-05
            seasonal_periods = 24
            if len(train) < seasonal_periods * 2:
                train = pd.concat(
                    [train, data[:train_start][len(train) - (seasonal_periods*2+1):-1]]).sort_index()

            train = np.log(train.fillna(train.mean()).applymap(lambda x: max(x, 1)) + eps)
            fit = ExponentialSmoothing(
                train.values,
                trend=None,
                seasonal='mul',
                seasonal_periods=seasonal_periods,
                damped_trend=False,
            ).fit()
            predict_index = data.index[data.index.get_loc(
                start) - 1:data.index.get_loc(end) + 2]
            nan_counts = len(data[start:end])
            predicted_df.loc[predict_index,
                             f'{title} - {col}: predict'] = np.exp(fit.forecast(int(nan_counts + 2)))

    return predicted_df


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
    initial_df = pd.read_excel(path, engine='openpyxl')
    title, data_groups, data_dfs = initialize_xslx_data(initial_df)
    resulted_df = pd.DataFrame(index=data_dfs[0][1].index)
    for name, df in data_dfs:
        predicted = predict_xslx(df, name)
        resulted_df = pd.concat([resulted_df, predicted], axis=1)

    resulted_df = resulted_df.reset_index()
    resulted_df = resulted_df.sort_index()
    resulted_df = resulted_df.rename(columns={'timestamp': 'Дата'})
    resulted_df['Дата'] = resulted_df['Дата'].astype(str)

    return initial_df, resulted_df
