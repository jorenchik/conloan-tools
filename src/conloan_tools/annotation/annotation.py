import click

from .make_sheet import make_sheet 
from .inject_suggestions import inject_suggestions 
from .translate import translate_target 
from .validate_sheet import validate
from .make_dataset import make_dataset 

@click.group()
def annotation():
    """Conloan annotation utilities."""

annotation.add_command(make_sheet)
annotation.add_command(inject_suggestions)
annotation.add_command(translate_target)
annotation.add_command(validate)
annotation.add_command(make_dataset)
