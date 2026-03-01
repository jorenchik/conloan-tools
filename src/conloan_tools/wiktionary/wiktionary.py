import click

from .extract_entries import extract
from .label_entries import get_lemmas 

@click.group()
def wiktionary():
    """Wiktionary extraction utilities."""

wiktionary.add_command(extract)
wiktionary.add_command(get_lemmas)
