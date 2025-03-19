import pandas as pd
import ast
import json
from typing import Union, Literal
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def initialize_data(path: str):
    puid_df = pd.read_csv(path)
    puid_df['keys'] = puid_df['keys'].apply(ast.literal_eval)
    puid_df['values'] = puid_df['values'].apply(ast.literal_eval)
    puid_df['meta'] = puid_df['meta'].apply(ast.literal_eval)
    meta = list(map(json.loads, puid_df['meta'][0]))
    titles = list(map(
        lambda x: f'{"Прямое" if  x["direction"] == 1 else "Обратное"}, полоса {x["lane"]}', meta))
    keys = puid_df['keys'][0][0]
    general_key_indices = [keys.index('volume'), keys.index(
        'speed'), keys.index('occupancy')]

    return puid_df


def get_na_intervals(dataframe: pd.DataFrame, year: int) -> pd.DataFrame:
    year_range = pd.date_range(
        f'{year}-01-01 00:00', f'{year}-12-31 23:55', freq='5Min')
    full_df = dataframe.reindex(year_range)
    full_df = full_df.reset_index()
    full_df['is_nan'] = full_df['volume'].isna().astype(int)
    full_df['group'] = (full_df['is_nan'].diff() == 1).cumsum()
    nan_intervals = full_df[full_df['is_nan'] == 1].groupby(
        'group').agg(start=('index', 'first'), end=('index', 'last'))
    nan_intervals['duration'] = nan_intervals['end'] - nan_intervals['start']
    return nan_intervals


def fill_na(dataframe: pd.DataFrame, nan_interval: (pd.Timestamp, pd.Timestamp, pd.Timedelta), method:  Literal['exp_smooth', 'moving_average']):
    start, end, duration = nan_interval
    if (method == 'exp_smooth'):
        # check type of interval
        # interval_len = pd.Timedelta()
        print(start, end, duration)


def process_dataframe(path: str):
    # df = initialize_data(path)
    df = pd.read_csv(path)
    nan_intervals = get_na_intervals(df, 2024)
    rows = list(nan_intervals.itertuples())
    nan_intervals.itertuples(lambda interval: fill_na(
        df, interval, method='exp_smooth'))


process_dataframe('test\input-data__19-03__23-28-55.csv')
