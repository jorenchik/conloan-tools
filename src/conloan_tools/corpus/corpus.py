import click

from .query import query 

@click.group()
def corpus():
    """Corpus utilities."""

corpus.add_command(query)
