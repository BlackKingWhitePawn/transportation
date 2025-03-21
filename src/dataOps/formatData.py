import pandas as pd
import ast
import json
from typing import Union, Literal
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def initialize_data(path: str, year: int):
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


def fill_na(df: pd.DataFrame, nan_interval: (pd.Timestamp, pd.Timestamp, pd.Timedelta), method:  Literal['exp_smooth', 'moving_average']):
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
            train['forecast'] = forecast
            forecast['volume'] = train['volume']
            pd.concat([train, forecast]).plot()

            pass
        else:
            pass


def process_dataframe(path: str):
    df = initialize_data(path, 2024)
    nan_intervals = get_na_intervals(df)

    def fill_interval(df, interval):
        start, end, duration = interval
        fill_na(df, interval, method='exp_smooth')

    for row in nan_intervals.itertuples():
        fill_na(df, row, method='exp_smooth')


process_dataframe('test\input-data__19-03__23-28-55.csv')
