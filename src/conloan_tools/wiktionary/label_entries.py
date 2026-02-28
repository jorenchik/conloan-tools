import csv
import argparse
import re
from collections import Counter

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
    # Note: 'cog' (cognate) and 'inh' (inherited) removed from here to focus
    # on loanwords for the table stats, though 'inh' was in the original map.
}

# Language code to full name mapping
LANGUAGE_NAMES = {
    "en": "English", "de": "German", "fr": "French", "it": "Italian",
    "es": "Spanish", "pt": "Portuguese", "la": "Latin", "grc": "Ancient Greek",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "is": "Icelandic",
    "fo": "Faroese", "nl": "Dutch", "mhn": "Middle High German",
    "lv": "Latvian", "lt": "Lithuanian", "pl": "Polish", "ru": "Russian",
    "cs": "Czech", "sk": "Slovak", "bg": "Bulgarian", "uk": "Ukrainian",
    "non": "Old Norse", "gem-pro": "Proto-Germanic",
    "ine-pro": "Proto-Indo-European",
    "sa": "Sanskrit", "hi": "Hindi", "fa": "Persian", "ar": "Arabic",
    "he": "Hebrew", "et": "Estonian", "fi": "Finnish", "hu": "Hungarian",
    "ga": "Irish", "cy": "Welsh", "br": "Breton", "sga": "Old Irish",
    "got": "Gotlandic", "goh": "Old High German", "osx": "Old Saxon",
    "ang": "Old English", "ofm": "Middle French", "fro": "Old French",
}


def extract_templates(text):
    """Extract all {{...}} templates and split by pipe."""
    pattern = r"\{\{([^}]+)\}\}"
    matches = re.findall(pattern, text)
    return [[elem.strip() for elem in match.split("|")] for match in matches]


def format_template(template):
    """Convert a template list into marked-up string."""
    if not template:
        return None

    t_type = template[0].lower()

    # Skip non-relevant templates
    if t_type in {
        "rfe", "suffix", "af", "inh", "cat", "考证", "zh-pron", "w", "cog"
    }:
        return None

    readable_prefix = TEMPLATE_READABLE.get(t_type)
    if not readable_prefix:
        return None

    # Handle template structures
    if t_type in TEMPLATE_READABLE:
        # {{type|lv|src_lang|word}}
        # Some templates might be short, handle gracefully
        if len(template) >= 2:
            # Check if index 2 exists (source lang), otherwise use 1
            lang_code = template[2] if len(template) > 2 else template[1]
            word = template[3] if len(template) > 3 else ""

            lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

            return f"<C>{readable_prefix}</C> <L>{lang_name}</L> {word}".strip()

    return None


def read_csv(csv_in, csv_out):
    stats = {
        "total_etymology": 0,
        "raw_bor_der_templates": 0,
        "filtered_bor_der": 0,
        "unique_filtered": 0,
    }
    
    counter = Counter()
    results = []
    word_set = set()

    # Set of templates that count towards the "bor/der" statistic
    LOAN_TEMPLATES = set(TEMPLATE_READABLE.keys())

    with open(csv_in, "r", encoding="utf-8", errors="ignore") as in_:
        reader = csv.DictReader(in_)

        for row in reader:
            word = row["word"]
            lang = row["language"].lower().strip()
            etymology = row["etymology_text"]

            if lang != "latvian":
                continue

            # 1. Total entries with etymology
            stats["total_etymology"] += 1

            # Extract templates early to check for "Raw" counts
            templates = extract_templates(etymology)
            
            # Check if this entry has ANY loanword-relevant templates (Raw Count)
            # This is done BEFORE filtering out uppercase/short words
            has_relevant_template = False
            for t in templates:
                if t and t[0].lower() in LOAN_TEMPLATES:
                    has_relevant_template = True
                    break
            
            if has_relevant_template:
                stats["raw_bor_der_templates"] += 1

            # --- START FILTERS ---

            if word and word[0].isupper():
                counter.update(["skipped_uppercase"])
                continue

            if word and len(word) <= 2:
                counter.update(["skipped_too_short"])
                continue

            if word and (word[0] == "-" or word[-1] == "-"):
                counter.update(["skipped_prefix_suffix"])
                continue

            if not templates:
                counter.update(["skipped_no_templates"])
                continue

            # Validate specific template arguments (lv recipient, non-proto)
            valid_templates = []
            for t in templates:
                t_type = t[0].lower()
                if t_type in LOAN_TEMPLATES:
                    # Wiktionary standard: {{bor|lv|...}}
                    # Check that the 2nd arg (recipient) is 'lv' if present
                    if len(t) > 1 and t[1] != "lv":
                        continue
                    # Exclude Proto-languages (often -pro suffix in code)
                    if len(t) > 2 and "-pro" in t[2].lower():
                        continue
                    valid_templates.append(t)

            if not valid_templates:
                counter.update(["skipped_no_valid_loan_templates"])
                continue

            # --- END FILTERS ---

            # If we are here, the entry is "Filtered" and valid
            stats["filtered_bor_der"] += 1
            counter.update(["processed"])

            # Format text for output CSV
            readable_entries = []
            for t in valid_templates:
                formatted = format_template(t)
                if formatted:
                    readable_entries.append(formatted)

            if not readable_entries:
                # Should not happen given valid_templates check, but safety net
                continue

            lang_info = "\n".join(readable_entries)

            # Check uniqueness
            if word not in word_set:
                word_set.add(word)
            
            results.append({
                "recepient_word": word,
                "donor_language": "",  # Placeholder
                "donor_word": "",      # Placeholder
                "lang_info": lang_info,
            })

    # Calculate final unique count
    stats["unique_filtered"] = len(word_set)

    # Write output
    with open(csv_out, "w", encoding="utf-8", newline="") as out:
        fieldnames = ["recepient_word", "donor_language", "donor_word", "lang_info"]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print Table Stats
    total = stats["total_etymology"]
    
    print("\n" + "="*50)
    print("STATISTIKA (Tabulai 4.1)")
    print("="*50)
    
    # 1. Total
    print(f"{'Satur etimoloģiju':<50} : {total}")
    print(f"{' ':50}   (100%)")
    
    # 2. Raw Templates
    count_1 = stats["raw_bor_der_templates"]
    perc_1 = (count_1 / total * 100) if total else 0
    print(f"{'bor vai der tipa veidne':<50} : {count_1}")
    print(f"{' ':50}   ({perc_1:.0f}%)")

    # 3. Filtered
    count_2 = stats["filtered_bor_der"]
    perc_2 = (count_2 / total * 100) if total else 0
    print(f"{'bor vai der tipa veidne (atfiltrētie)':<50} : {count_2}")
    print(f"{' ':50}   ({perc_2:.0f}%)")

    # 4. Unique
    count_3 = stats["unique_filtered"]
    perc_3 = (count_3 / total * 100) if total else 0
    print(f"{'bor vai der tipa veidne (atfiltrētie un unikālie)':<50} : {count_3}")
    print(f"{' ':50}   ({perc_3:.0f}%)")
    
    print("\n" + "="*50)
    print("FILTRU KOPSAVILKUMS (Debug)")
    print("="*50)
    for item, count in counter.most_common():
        print(f"{item:30s}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_in", help="Input CSV file path")
    parser.add_argument("csv_out", help="Output CSV file path")
    args = parser.parse_args()

    read_csv(args.csv_in, args.csv_out)


if __name__ == "__main__":
    main()
