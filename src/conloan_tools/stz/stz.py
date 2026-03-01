import click

from .lemmatize import lemmatize 

@click.group()
def stz():
    """Stanza utilities."""

stz.add_command(lemmatize)
