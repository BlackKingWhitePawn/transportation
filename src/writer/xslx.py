import pandas as pd
import logging

logger = logging.getLogger(__name__)


def write_to_predict_sheet(path, df):
    with pd.ExcelWriter(path, mode='a', if_sheet_exists='replace') as writer:
        logger.info('Writing to Excel - create/update "predict" sheet')
        try:
            df.to_excel(excel_writer=writer,
                        sheet_name='Результаты',
                        index=False)
        except:
            logger.error('Writing to Excel crashed')
