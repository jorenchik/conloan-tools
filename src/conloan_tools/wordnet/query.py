import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import click

from conloan_tools.wordnet import wordnet


@dataclass(frozen=True)
class Lemma:
    written_form: str
    part_of_speech: str


@dataclass
class LexicalEntry:
    id: str
    lemma: Lemma
    senses: Dict[str, str] = field(default_factory=dict)


@dataclass
class Synset:
    id: str
    definition: Optional[str] = None
    members: List[str] = field(default_factory=list)


@dataclass
class SenseMatch:
    synset_id: str
    definition: Optional[str]
    synonyms: List[str]


@dataclass
class EntryMatch:
    pos: str
    senses: List[SenseMatch]


@dataclass
class SearchResult:
    word: str
    found: bool
    entries: List[EntryMatch]


class WordNet:
    """In-memory representation of a WordNet LMF XML file,
    indexed for fast lemma-based lookup."""

    def __init__(self, file_path: str):
        self.entries: Dict[str, LexicalEntry] = {}
        self.synsets: Dict[str, Synset] = {}
        self._word_to_entries: Dict[str, List[str]] = {}
        self._load_from_lmf(file_path)

    def _load_from_lmf(self, file_path: str) -> None:
        """Stream-parse an LMF XML file, populating entries and
        synsets while keeping peak memory low via incremental
        element clearing."""
        context = ET.iterparse(file_path, events=("end",))
        for _, elem in context:
            if elem.tag == "LexicalEntry":
                self._ingest_entry(elem)
                elem.clear()
            elif elem.tag == "Synset":
                self._ingest_synset(elem)
                elem.clear()

    def _ingest_entry(self, elem: ET.Element) -> None:
        """Extract a LexicalEntry (lemma + senses) from an XML
        element and register it in the lookup indices."""
        entry_id = elem.get("id")
        lemma_elem = elem.find("Lemma")
        if not entry_id or lemma_elem is None:
            return

        lemma = Lemma(
            written_form=lemma_elem.get("writtenForm", ""),
            part_of_speech=lemma_elem.get("partOfSpeech", ""),
        )
        entry = LexicalEntry(id=entry_id, lemma=lemma)

        for sense in elem.findall("Sense"):
            s_id = sense.get("id")
            syn_id = sense.get("synset")
            if s_id and syn_id:
                entry.senses[s_id] = syn_id

        self.entries[entry_id] = entry
        word_key = lemma.written_form.lower()
        self._word_to_entries.setdefault(word_key, []).append(entry_id)

    def _ingest_synset(self, elem: ET.Element) -> None:
        """Extract a Synset (definition + member list) from an XML
        element and store it."""
        syn_id = elem.get("id")
        if not syn_id:
            return

        members_attr = elem.get("members", "")
        members_list = members_attr.split() if members_attr else []
        defn_elem = elem.find("Definition")
        defn_text = defn_elem.text if defn_elem is not None else None

        self.synsets[syn_id] = Synset(
            id=syn_id, definition=defn_text, members=members_list
        )

    def get_synonym_groups(self, word: str) -> SearchResult:
        """Look up a word and return every sense it participates in,
        together with co-members of each synset (i.e. synonyms)."""
        query = word.strip().lower()
        e_ids = self._word_to_entries.get(query, [])

        if not e_ids:
            return SearchResult(word=word, found=False, entries=[])

        entry_matches: List[EntryMatch] = []
        for eid in e_ids:
            entry = self.entries[eid]
            sense_matches: List[SenseMatch] = []

            for syn_id in entry.senses.values():
                syn = self.synsets.get(syn_id)
                if not syn:
                    continue

                synonym_names: List[str] = []
                for member_id in syn.members:
                    member_entry = self.entries.get(member_id)
                    if member_entry:
                        name = member_entry.lemma.written_form
                        if name.lower() != query:
                            synonym_names.append(name)

                sense_matches.append(
                    SenseMatch(
                        synset_id=syn_id,
                        definition=syn.definition,
                        synonyms=sorted(set(synonym_names)),
                    )
                )

            entry_matches.append(
                EntryMatch(pos=entry.lemma.part_of_speech, senses=sense_matches)
            )

        return SearchResult(word=word, found=True, entries=entry_matches)


def print_result(result: SearchResult) -> None:
    """Pretty-print a SearchResult to stdout."""
    if not result.found:
        click.echo(
            click.style(
                f"Error: '{result.word}' not found.", fg="red"
            )
        )
        return

    click.echo(
        f"\nWord '{result.word}' found "
        f"({len(result.entries)} variant[s])."
    )

    for entry in result.entries:
        click.echo(f"\n--- Part of Speech: {entry.pos.upper()} ---")
        if not entry.senses:
            click.echo("    No semantic senses defined for this variant.")
            continue

        for sense in entry.senses:
            click.echo(
                f"\nMeaning: {sense.definition or 'No definition available'}"
            )
            if sense.synonyms:
                click.echo(f"Synonyms: {', '.join(sense.synonyms)}")
            else:
                click.echo(
                    "Synonyms: No other synonyms linked to this meaning."
                )

    click.echo("\n" + "-" * 30)


@click.command("query")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "-w",
    "--word",
    default=None,
    help="Single word to look up. Omit for interactive mode.",
)
def query(file: str, word: Optional[str]) -> None:
    """Tezaurs WordNet LMF synonym lookup.

    Loads FILE (an LMF XML export) and either performs a single
    lookup (--word) or enters an interactive REPL."""
    click.echo(f"Loading {file}...")
    wn = WordNet(file)
    click.echo(
        f"Loaded {len(wn.entries)} entries and "
        f"{len(wn.synsets)} synsets."
    )

    if word:
        print_result(wn.get_synonym_groups(word))
        return

    # Interactive mode
    click.echo("\n" + "=" * 50)
    click.echo("Latvian WordNet Synonym Search")
    click.echo("Type 'exit' to quit.")
    click.echo("=" * 50)

    while True:
        try:
            q = click.prompt("\nWord to search", default="", show_default=False)
            q = q.strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            print_result(wn.get_synonym_groups(q))
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    wordnet()
