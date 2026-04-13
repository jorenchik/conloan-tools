import click

from .csv_extract import csv_extract


@click.group("utils")
def utils():
    """General-purpose utilities."""


utils.add_command(csv_extract)
