import click

from .interactive import interactive 

@click.group()
def translate():
    """Corpus utilities."""

translate.add_command(interactive)
