import click
from .train import train, kfold, eval_cmd

@click.group()
def classifier():
    """Conloan classifier utilities."""

classifier.add_command(train)
classifier.add_command(kfold)
classifier.add_command(eval_cmd)
