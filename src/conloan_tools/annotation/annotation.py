import click

from conloan_tools.annotation.sheet.make_from_lemmas import make_from_lemmas 
from conloan_tools.annotation.sheet.translate import translate_sheet
from conloan_tools.annotation.sheet.validate_sheet import validate
from conloan_tools.annotation.sheet.replacement import generate_repl_placeholders 
from conloan_tools.annotation.sheet.assistant import assistant
from conloan_tools.annotation.json.make_from_sheet import make_from_sheet 

@click.group()
def annotation():
    """Conloan annotation utilities."""

@annotation.group()
def sheet():
    """Conloan annotation sheet utilities."""

sheet.add_command(make_from_lemmas)
sheet.add_command(validate)
sheet.add_command(translate_sheet)
sheet.add_command(generate_repl_placeholders)
sheet.add_command(assistant)

@annotation.group()
def json():
    """Conloan dataset json utilities."""

json.add_command(make_from_sheet)
