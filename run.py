import click

@click.group()
def main():
    pass


@main.command()
@click.option('--fill-na', default=None, type=str, help='CSV file to process')
@click.option('--format', default='x', type=str, help='Data format which should be processing. [x] for xologie data, [c] for cusadd')
def data_operations(fill_na):
    click.echo(fill_na)


if __name__ == "__main__":
    main()