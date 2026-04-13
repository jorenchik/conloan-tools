import click

from conloan_tools.wiktionary import wiktionary
from conloan_tools.corpus import corpus 
from conloan_tools.wordnet import wordnet 
from conloan_tools.annotation import annotation 
from conloan_tools.stz import stz 
from conloan_tools.translate import translate 
from conloan_tools.wb import wb 
from conloan_tools.ner import ner
from conloan_tools.classifier import classifier
from conloan_tools.utils import utils 

@click.group()
def cli():
    """conloan-tools CLI."""

cli.add_command(wiktionary)
cli.add_command(corpus)
cli.add_command(wordnet)
cli.add_command(annotation)
cli.add_command(stz)
cli.add_command(translate)
cli.add_command(wb)
cli.add_command(ner)
cli.add_command(classifier)
cli.add_command(utils)

if __name__ == "__main__":
    cli()
