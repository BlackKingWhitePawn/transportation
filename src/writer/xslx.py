import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def write_to_predict_sheet(path: Path, initial_df, predicted):
    with pd.ExcelWriter( f'{Path.joinpath(path.parent,"Обработано__" + path.stem + ".xlsx")}', engine='xlsxwriter', datetime_format="yyyy-mm-dd HH:MM:SS") as writer:
        logger.info('Writing to Excel - create/update "predict" sheet')
        try:
            initial_df.to_excel(excel_writer=writer, sheet_name='Исходные данные', merge_cells=True)
            predicted = predicted.apply(pd.to_numeric, errors='coerce').round().astype('Int64')
            predicted.to_excel(excel_writer=writer, sheet_name='Результаты', merge_cells=True)
            predicted.combine_first(initial_df).to_excel(excel_writer=writer, sheet_name='Объединенные данные', merge_cells=True)
            
            date_format = writer.book.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss'})
            for sheet in writer.sheets:
                writer.sheets[sheet].set_column('A:A', 20, date_format)  
        except:
            logger.error('Writing to Excel crashed')
