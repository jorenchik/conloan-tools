import re
import csv
import argparse
from tqdm import tqdm
import os
import sys

LANG_HEADER    = re.compile(r"(^|[^=])==([A-Za-z ]+)==")
HEADER         = re.compile(r"===([A-Za-z ]+)===")
SECTION_HEADER = re.compile(r"^===")
TITLE_TAG        = re.compile(r"<title>(.*?)</title>")
SPECIFIER = re.compile(r"\{\{[^}]+\}\}", re.IGNORECASE)

# Default if no file is provided
DEFAULT_LANGS = {"Latvian"}

def extract_wiktionary(
    filename: str,
    csv_out: str,
    allowed_languages: set[str],
    include_etymology: bool = True,
    range_start: int = None,
    range_end: int = None,
    total_lines: int = None,
):
    buffer = []
    
    # State tracking
    inside_allowed_lang = False
    inside_etym = False
    
    # Data tracking
    word = None
    lang = None
    
    # Helper to calculate total lines if not provided
    if not total_lines:
        print("Counting lines in the input file...")
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            total_lines = sum(1 for _ in f)

    with (
        open(filename, "r", encoding="utf-8", errors="ignore") as f,
        open(csv_out, "w", encoding="utf-8", newline="") as out
    ):
        writer = csv.writer(out)
        # We keep the header consistent, but etymology_text will be empty if disabled
        headers = ["line", "language", "word"]
        if include_etymology:
            headers.append("etymology_text")
        writer.writerow(headers)

        for line_num, line in tqdm(enumerate(f, start=1), total=total_lines):
            if range_start and line_num < range_start:
                continue
            if range_end and line_num > range_end:
                break

            # Check for Headers (Language or Title)
            is_lang_header = LANG_HEADER.search(line)
            is_title_tag = TITLE_TAG.search(line)

            if is_lang_header or is_title_tag:
                # 1. Flush previous buffer if valid
                # We check the *previous* state variables here
                if inside_allowed_lang and word and lang:
                    row = [line_num, lang, word]
                    
                    if include_etymology:
                        normalized_text = ""
                        if buffer:
                            normalized_text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
                        if normalized_text:
                            row.append(normalized_text)
                            writer.writerow(row)
                    else:
                        writer.writerow(row)

                # 2. Reset / Update State
                buffer.clear()
                inside_etym = False

                if is_title_tag:
                    res = TITLE_TAG.search(line)
                    word = res.group(1)
                    # Title tag resets language context in XML dumps usually
                    inside_allowed_lang = False 
                    lang = None

                if is_lang_header:
                    res = LANG_HEADER.search(line)
                    lang = res.group(2)
                    inside_allowed_lang = lang in allowed_languages
                
                continue

            # Check for Section Headers (Etymology, Noun, Verb, etc.)
            res = HEADER.search(line)
            if res:
                name = res.group(1)
                # Only enter etymology mode if we are in a target lang AND user wants etymology
                if inside_allowed_lang and include_etymology:
                    if name == "Etymology":
                        inside_etym = True
                        continue
                    else:
                        inside_etym = False
            
            # Capture Content
            if inside_etym and inside_allowed_lang:
                buffer.append(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml")
    parser.add_argument("--out", default="extracted.csv")
    
    # (1) Language inputs
    parser.add_argument(
        "--languages-file", 
        type=str, 
        help="Path to .txt file with one language per line. Defaults to Latvian if omitted."
    )

    # (2) Etymology toggle
    parser.add_argument(
        "--include-etymology",
        action="store_true",
        default=False,
        help="If set, parses and includes etymology text. Otherwise column is empty."
    )

    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--total-lines", type=int, default=None)

    args = parser.parse_args()

    if os.path.exists(args.out):
        confirm = input(f"File {args.out} already exists, overwrite? [y/N] ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            sys.exit(0)

    # Determine allowed languages
    allowed_languages = DEFAULT_LANGS.copy()
    if args.languages_file:
        if not os.path.exists(args.languages_file):
            print(f"Error: Language file {args.languages_file} not found.")
            sys.exit(1)
        with open(args.languages_file, "r", encoding="utf-8") as f:
            # Create set, strip whitespace, ignore empty lines
            allowed_languages = {line.strip() for line in f if line.strip()}
            print(f"Loaded {len(allowed_languages)} allowed languages.")

    extract_wiktionary(
        args.xml,
        args.out,
        allowed_languages=allowed_languages,
        include_etymology=args.include_etymology,
        range_start=args.start,
        range_end=args.end,
        total_lines=args.total_lines,
    )

if __name__ == "__main__":
    main()
