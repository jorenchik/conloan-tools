import click

from .nt_translate import interactive 

@click.group()
def translate():
    """Corpus utilities."""

translate.add_command(interactive)
