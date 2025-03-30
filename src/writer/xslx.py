import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def write_to_predict_sheet(path: Path, initial_df, resulted_df):
    with pd.ExcelWriter(f'{Path.joinpath(path.parent,  path.stem + "-processed.xlsx")}', engine='xlsxwriter') as writer:
        logger.info('Writing to Excel - create/update "predict" sheet')
        try:
            # initial data
            initial_df.to_excel(excel_writer=writer,
                                 sheet_name='Исходные данные')
            # resulted
            numeric_cols = resulted_df.select_dtypes(include='float').columns
            resulted_df[numeric_cols] = resulted_df[numeric_cols].round(0)
            resulted_df = resulted_df.reset_index().rename(columns={'timestamp': 'Дата'})
            resulted_df['Дата'] = resulted_df['Дата'].astype(str)
            resulted_df.to_excel(excel_writer=writer,
                                 sheet_name='Результаты')
            # merged
            d_initial = initial_df[4:-3].copy().reset_index(drop=True)
            if ('index' in resulted_df.columns):
                resulted_df.drop('index', axis=1, inplace=True)
            d_initial.columns = resulted_df.columns
            filled = resulted_df.combine_first(d_initial)
            filled.to_excel(excel_writer=writer,
                                 sheet_name='Объединенные данные')
        except:
            logger.error('Writing to Excel crashed')
