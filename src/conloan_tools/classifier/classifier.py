import click

from .eval import evaluate 
from .train import train

@click.group()
def classifier():
    """Conloan classifier utilities."""

classifier.add_command(evaluate)
classifier.add_command(train)
