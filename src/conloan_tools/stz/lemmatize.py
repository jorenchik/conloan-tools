"""Stanza-based lemmatization with canonical language name resolution."""
import os
import click

from conloan_tools.resources import load_known_languages


def resolve_language_code(language: str) -> str:
    """Map a canonical language name (e.g. 'Latvian') to a Stanza code."""
    known = load_known_languages()
    name_to_code = {name.lower(): code for code, name in known.items()}
    key = language.strip().lower()
    if key not in name_to_code:
        available = ", ".join(sorted(known.values()))
        raise ValueError(
            f"Unknown language '{language}'. Available: {available}"
        )
    return name_to_code[key]


class Lemmatizer:
    """Single-language Stanza lemmatizer (lazy-loaded)."""

    def __init__(self, language: str, *, verbose: bool = False):
        """
        Parameters
        ----------
        language : str
            Canonical language name (e.g. ``'Latvian'``).
        """
        self.code = resolve_language_code(language)
        self._verbose = verbose
        self._nlp = None
        self._ensure_pipeline()

    def _ensure_pipeline(self):
        if self._nlp is not None:
            return
        import stanza

        stanza.download(
            self.code,
            processors="tokenize,lemma",
            logging_level="ERROR",
        )
        self._nlp = stanza.Pipeline(
            self.code,
            processors="tokenize,lemma",
            verbose=self._verbose,
            logging_level="ERROR",
        )

    def lemmatize(self, text: str) -> list[str]:
        """Return all lemmas for every word in *text*."""
        self._ensure_pipeline()
        doc = self._nlp(text)
        return [
            word.lemma
            for sent in doc.sentences
            for word in sent.words
        ]

    def lemmatize_word(self, word: str) -> str:
        """Return the lemma of a single word token."""
        lemmas = self.lemmatize(word)
        return lemmas[0] if lemmas else word

    @property
    def model_identifier(self):
        """Returns the filenames for the tokenizer and lemmatizer models."""
        if not self._nlp:
            return None
        
        config = self._nlp.config
        # Extract filenames from the full system paths
        tokenizer = os.path.basename(config.get("tokenize_model_path", "N/A"))
        lemmatizer = os.path.basename(config.get("lemma_model_path", "N/A"))
        
        return f"tokenizer:{tokenizer} lemmatizer:{lemmatizer}"


@click.command("lemmatize")
@click.option(
    "--language",
    required=True,
    help="Canonical language name (e.g. 'Latvian')",
)
@click.argument("words", nargs=-1)
def lemmatize(language, words):
    """Lemmatize words or start an interactive session.

    If WORDS are given, print their lemmas and exit.
    Otherwise, enter an interactive REPL.
    """
    lem = Lemmatizer(language)

    if words:
        for w in words:
            click.echo(f"{w} → {lem.lemmatize_word(w)}")
        return

    click.echo(
        "Interactive mode. Enter text to lemmatize (Ctrl-D to quit)."
    )
    while True:
        try:
            text = click.prompt("", prompt_suffix="> ")
        except (EOFError, click.Abort):
            break
        if not text.strip():
            continue
        lemmas = lem.lemmatize(text)
        click.echo(" ".join(lemmas))
