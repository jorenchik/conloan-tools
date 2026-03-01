import csv
import re
import os
from collections import Counter

import click

from conloan_tools.resources import load_known_languages
from conloan_tools.wiktionary import wiktionary 

# Mapping of Wiktionary template codes to human-readable text
TEMPLATE_READABLE = {
    "bor": "borrowed from",
    "lbor": "learned borrowing from",
    "mbor": "possibly borrowed from",
    "der": "derived from",
    "aff": "affix from",
    "suff": "suffix from",
    "pref": "prefix from",
    "inch": "inherited from",
    "relbor": "related borrowing from",
}

SKIP_TEMPLATES = {
    "rfe", "suffix", "af", "inh", "cat", "考证", "zh-pron", "w", "cog"
}

LOAN_TEMPLATES = set(TEMPLATE_READABLE.keys())


def extract_templates(text):
    """Extract all {{...}} templates and split by pipe."""
    pattern = r"\{\{([^}]+)\}\}"
    matches = re.findall(pattern, text)
    return [[elem.strip() for elem in match.split("|")] for match in matches]


def format_template(template, language_names):
    """Convert a template list into marked-up string."""
    if not template:
        return None

    t_type = template[0].lower()

    if t_type in SKIP_TEMPLATES:
        return None

    readable_prefix = TEMPLATE_READABLE.get(t_type)
    if not readable_prefix:
        return None

    if len(template) >= 2:
        lang_code = template[2] if len(template) > 2 else template[1]
        word = template[3] if len(template) > 3 else ""
        lang_name = language_names.get(lang_code, lang_code)
        return f"<C>{readable_prefix}</C> <L>{lang_name}</L> {word}".strip()

    return None


def has_any_loan_template(templates):
    """Check if any template is a loanword-relevant type."""
    return any(t and t[0].lower() in LOAN_TEMPLATES for t in templates)


def extract_valid_loan_templates(templates):
    """Filter templates to those with lv recipient and non-proto source."""
    valid = []
    for t in templates:
        t_type = t[0].lower()
        if t_type not in LOAN_TEMPLATES:
            continue
        if len(t) > 1 and t[1] != "lv":
            continue
        if len(t) > 2 and "-pro" in t[2].lower():
            continue
        valid.append(t)
    return valid


@click.command("get-lemmas")
@click.argument("csv_in", type=click.Path(exists=True))
@click.argument("csv_out", type=click.Path())
def get_lemmas(csv_in, csv_out):

    if os.path.exists(csv_out):
        click.confirm(f"Overwrite {csv_out}?", abort=True)

    """Extract Latvian loanword lemmas from Wiktionary etymology dump."""
    language_names = load_known_languages()

    # Ordered pipeline counters
    stage = Counter()
    excluded_by = Counter()

    results = []
    word_set = set()

    with open(csv_in, "r", encoding="utf-8", errors="ignore") as in_:
        reader = csv.DictReader(in_)

        for row in reader:
            word = row["word"]
            lang = row["language"].lower().strip()
            etymology = row["etymology_text"]

            if lang != "latvian":
                continue

            stage["1_total_etymology"] += 1

            templates = extract_templates(etymology)

            if has_any_loan_template(templates):
                stage["2_raw_has_loan_template"] += 1

            # --- FILTER: uppercase ---
            if word and word[0].isupper():
                excluded_by["uppercase_initial"] += 1
                continue

            stage["3_after_uppercase"] += 1

            # --- FILTER: too short ---
            if word and len(word) <= 2:
                excluded_by["too_short"] += 1
                continue

            stage["4_after_length"] += 1

            # --- FILTER: affix form ---
            if word and (word[0] == "-" or word[-1] == "-"):
                excluded_by["affix_form"] += 1
                continue

            stage["5_after_affix"] += 1

            # --- FILTER: no templates at all ---
            if not templates:
                excluded_by["no_templates"] += 1
                continue

            stage["6_after_has_templates"] += 1

            # --- FILTER: no valid loan templates ---
            valid_templates = extract_valid_loan_templates(templates)

            if not valid_templates:
                excluded_by["no_valid_loan_template"] += 1
                continue

            stage["7_after_valid_loan"] += 1

            # --- PASSED ALL FILTERS ---
            readable_entries = []
            for t in valid_templates:
                formatted = format_template(t, language_names)
                if formatted:
                    readable_entries.append(formatted)

            if not readable_entries:
                excluded_by["format_returned_none"] += 1
                continue

            stage["8_after_format"] += 1

            if word not in word_set:
                word_set.add(word)

            results.append({
                "word": word,
                "valid": "",
                "info": "\n".join(readable_entries),
            })

    stage["9_unique_words"] = len(word_set)

    # --- Write output ---
    with open(csv_out, "w", encoding="utf-8", newline="") as out:
        fieldnames = ["word", "valid", "info"]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # --- Print statistics ---
    total = stage["1_total_etymology"]

    def pct(n):
        return f"({n / total * 100:.1f}%)" if total else "(0%)"

    click.echo("")
    click.echo("=" * 60)
    click.echo("PIPELINE STAGES")
    click.echo("=" * 60)

    labels = [
        ("1_total_etymology", "Latvian entries with etymology"),
        ("2_raw_has_loan_template", "  ↳ of which have loan template (raw)"),
        ("3_after_uppercase", "After excluding uppercase"),
        ("4_after_length", "After excluding len ≤ 2"),
        ("5_after_affix", "After excluding affix forms"),
        ("6_after_has_templates", "After excluding no-template entries"),
        ("7_after_valid_loan", "After excluding no valid loan template"),
        ("8_after_format", "After formatting check"),
        ("9_unique_words", "Unique words in output"),
    ]

    for key, label in labels:
        n = stage[key]
        click.echo(f"  {label:<50} {n:>6}  {pct(n)}")

    click.echo("")
    click.echo("=" * 60)
    click.echo("EXCLUSIONS BY FILTER")
    click.echo("=" * 60)

    for reason, count in excluded_by.most_common():
        click.echo(f"  {reason:<50} {count:>6}")

    click.echo("")


if __name__ == "__main__":
    wiktionary()
