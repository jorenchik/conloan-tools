import csv
import io

from importlib.resources import files

_WIKTIONARY_PACKAGE = "conloan_tools.wiktionary"

def _read_resource(package: str, filename: str) -> str:
    return (
        files(package)
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


def load_known_languages() -> dict[str, str]:
    text = _read_resource(_WIKTIONARY_PACKAGE, "wiktionary_languages.csv")
    stream = io.StringIO(text)
    reader = csv.DictReader(stream)
    return {row["code"]: row["canonical_name"] for row in reader}
