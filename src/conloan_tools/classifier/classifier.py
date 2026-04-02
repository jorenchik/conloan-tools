import click
from .train import train, kfold, eval_cmd, inspect_tokens, inspect_predictions

@click.group()
def classifier():
    """Conloan classifier utilities."""

classifier.add_command(train)
classifier.add_command(kfold)
classifier.add_command(eval_cmd)
classifier.add_command(inspect_tokens)
classifier.add_command(inspect_predictions)
