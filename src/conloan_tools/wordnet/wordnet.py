import click

from .query import query

@click.group()
def wordnet():
    """Wordnet utilities."""

wordnet.add_command(query)
