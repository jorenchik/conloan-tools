import click

from .make_sheet import make_sheet 
from .translate import translate_target 
from .validate_sheet import validate
from .make_dataset import make_dataset 

@click.group()
def annotation():
    """Conloan annotation utilities."""

@annotation.group()
def sheet():
    """Conloan annotation sheet utilities."""

sheet.add_command(make_sheet)
sheet.add_command(validate)
sheet.add_command(translate_target)

@annotation.group()
def json():
    """Conloan dataset json utilities."""

json.add_command(make_dataset)
