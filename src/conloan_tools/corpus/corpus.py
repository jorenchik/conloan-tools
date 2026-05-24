import click

from .query import query_group
from .build_index import build_index, convert_index
from .inspect_index import inspect_index

@click.group()
def corpus():
    """Conloan corpus utilities."""

corpus.add_command(query_group)
corpus.add_command(build_index)
corpus.add_command(convert_index)
corpus.add_command(inspect_index)
