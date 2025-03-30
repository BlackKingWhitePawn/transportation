import click
import src.dataOps as dataOps
import src.writer as writer
import logging
import os
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


@click.group()
def main():
    logging.basicConfig(filename='log.txt', level=logging.INFO)
    pass


@main.command()
@click.argument('path', type=str)
@click.option('--format', default='cd', type=str, help='''Формат данных. 
\n"xg" - сырые данные из иксолоджи. Ожидаются csv файлы
\n"cd" - экспорт из ЦУСАД. Ожидаются Эксель файлы. Обрабатываться будет первый лист файла
''')
def data_operations(path, format):
    """
Обрабатывает входные данные, заполняя пропуски.
На вход принимает путь к файлу с данными или к директории.
Если передан путь к директории то все xls/xlsx/csv в директории будут обработаны.
При обработке Эксель файлов результат записывается в новый лист "Результаты".
Примеры:
\nrun data-operations  \"C:\\Users\\riabt\\Downloads\"
\nrun data-operations \"C:\\Users\\riabt\\Download\\Telegram Desktop\" --format cd
"""
    extensions = {".xls",  ".xlsx"} if (format == 'cd') else {".csv"}
    path_object = Path(path)

    def process_file(path: Path):
        if (format == 'xg'):
            dataOps.process_csv_dataframe(path)
        elif (format == 'cd'):
            print(f'Обработка {path}')
            initial_df, resulted_df = dataOps.process_xslx_dataframe(path)
            print(f'Запись в файл {path}')
            writer.write_to_predict_sheet(path, initial_df, resulted_df)

    if (path_object.is_dir()):
        for filePath in tqdm([f for f in path_object.iterdir() if f.is_file() and f.suffix in extensions]):
            process_file(filePath)


if __name__ == "__main__":
    main()
