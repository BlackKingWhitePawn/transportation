import click
import src.dataOps as dataOps
import src.writer as writer
import logging

logger = logging.getLogger(__name__)


@click.group()
def main():
    logging.basicConfig(filename='log.txt', level=logging.INFO)
    pass


@main.command()
@click.option('--path', default=None, type=str, help='CSV file to process')
@click.option('--format', default='x', type=str, help='Data format which should be processing. [x] for xologie data, [c] for cusadd')
def data_operations(path, format):
    print(path)
    if (format == 'x'):
        dataOps.process_csv_dataframe('test\input-data__19-03__23-28-55.csv')
    elif (format == 'c'):
        df = dataOps.process_xslx_dataframe(path)
        writer.write_to_predict_sheet(path, df)


if __name__ == "__main__":
    main()
