import re
import csv
import os

import click
from tqdm import tqdm

from conloan_tools.resources import load_known_languages
from conloan_tools.wiktionary import wiktionary

LANG_HEADER = re.compile(r"(^|[^=])==([A-Za-z ]+)==")
HEADER = re.compile(r"===([A-Za-z ]+)===")
SECTION_HEADER = re.compile(r"^===")
TITLE_TAG = re.compile(r"<title>(.*?)</title>")
SPECIFIER = re.compile(r"\{\{[^}]+\}\}", re.IGNORECASE)


def extract_wiktionary(
    filename: str,
    csv_out: str,
    allowed_languages: set[str],
    mode: str = "etymology",
    drop_etymology: bool = False,
    range_start: int | None = None,
    range_end: int | None = None,
):
    buffer = []

    inside_allowed_lang = False
    inside_etym = False

    word = None
    lang = None

    file_size = os.path.getsize(filename)

    with (
        open(filename, "r", encoding="utf-8", errors="ignore") as f,
        open(csv_out, "w", encoding="utf-8", newline="") as out,
    ):
        writer = csv.writer(out)
        headers = ["line", "language", "word"]
        if not drop_etymology:
            headers.append("etymology_text")
        writer.writerow(headers)

        prev_pos = 0
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for line_num, line in enumerate(f, start=1):
                pos = f.buffer.tell()
                pbar.update(pos - prev_pos)
                prev_pos = pos

                if range_start and line_num < range_start:
                    continue
                if range_end and line_num > range_end:
                    break

                is_lang_header = LANG_HEADER.search(line)
                is_title_tag = TITLE_TAG.search(line)

                if is_lang_header or is_title_tag:
                    if inside_allowed_lang and word and lang:
                        normalized_text = ""
                        if buffer:
                            normalized_text = re.sub(
                                r"\s+", " ", " ".join(buffer)
                            ).strip()

                        should_write = mode == "all" or normalized_text

                        if should_write:
                            row = [line_num, lang, word]
                            if not drop_etymology:
                                row.append(normalized_text)
                            writer.writerow(row)

                    buffer.clear()
                    inside_etym = False

                    if is_title_tag:
                        res = TITLE_TAG.search(line)
                        word = res.group(1)
                        inside_allowed_lang = False
                        lang = None

                    if is_lang_header:
                        res = LANG_HEADER.search(line)
                        lang = res.group(2)
                        inside_allowed_lang = lang in allowed_languages

                    continue

                res = HEADER.search(line)
                if res:
                    name = res.group(1)
                    if inside_allowed_lang:
                        if name == "Etymology":
                            inside_etym = True
                            continue
                        else:
                            inside_etym = False

                if inside_etym and inside_allowed_lang:
                    buffer.append(line)


def validate_languages(allowed_languages: set[str]) -> None:
    try:
        # returns dict[code, name]
        known_map = load_known_languages()
    except FileNotFoundError:
        click.secho(
            "wiktionary_languages.csv not found — skipping validation. "
            "Ensure the resource exists.",
            fg="yellow",
        )
        return

    # Extract names for set-based validation
    known_names = set(known_map.values())
    
    unknown = allowed_languages - known_names
    if unknown:
        raise click.BadParameter(
            f"Unrecognized language(s): {', '.join(sorted(unknown))}. "
            "Check spelling or ensure the CSV is up to date.",
            param_hint="--languages",
        )


@click.command("extract")
@click.argument("xml")
@click.option("--out", default="extracted.csv")
@click.option(
    "--languages",
    type=str,
    required=True,
    help="Comma-separated list of languages.",
)
@click.option(
    "--mode",
    type=click.Choice(["etymology", "all"], case_sensitive=False),
    default="etymology",
    help=(
        "'etymology' emits only entries with etymology text; "
        "'all' emits every entry. Default: etymology."
    ),
)
@click.option(
    "--drop-etymology",
    is_flag=True,
    default=False,
    help="Omit the etymology_text column entirely.",
)
@click.option("--start", type=int, default=None)
@click.option("--end", type=int, default=None)
def extract(
    xml: str,
    out: str,
    languages: str,
    mode: str,
    drop_etymology: bool,
    start: int | None,
    end: int | None,
):
    """Extract entries from a Wiktionary XML dump."""
    if os.path.exists(csv_out):
        click.confirm(f"Overwrite {out}?", abort=True)

    allowed_languages = {
        l.strip() for l in languages.split(",") if l.strip()
    }
    if not allowed_languages:
        raise click.BadParameter(
            "No valid languages provided.", param_hint="--languages"
        )
    validate_languages(allowed_languages)
    click.echo(f"Loaded {len(allowed_languages)} allowed language(s).")

    extract_wiktionary(
        xml,
        out,
        allowed_languages=allowed_languages,
        mode=mode,
        drop_etymology=drop_etymology,
        range_start=start,
        range_end=end,
    )


if __name__ == "__main__":
    wiktionary()
