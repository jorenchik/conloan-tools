import click

from conloan_tools.wiktionary import wiktionary
from conloan_tools.corpus import corpus 
from conloan_tools.wordnet import wordnet 
from conloan_tools.annotation import annotation 
from conloan_tools.stz import stz 
from conloan_tools.translate import translate 

@click.group()
def cli():
    """conloan-tools CLI."""

cli.add_command(wiktionary)
cli.add_command(corpus)
cli.add_command(wordnet)
cli.add_command(annotation)
cli.add_command(stz)
cli.add_command(translate)

if __name__ == "__main__":
    cli()
