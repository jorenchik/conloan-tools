
import math
import re
from collections import defaultdict
from tqdm import tqdm
import pickle
import functools
import click
import numpy as np
import os

from conloan_tools.wb import wb 

# ----- Language model -----

class WittenBellCharLM:
    """Char n-gram LM with Witten-Bell smoothing.

    Counts are stored as numpy int64 arrays keyed by context string,
    one array of shape (vocab_size,) per context.  This replaces the
    nested-defaultdict approach and cuts memory ~3-4x.
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.char2id: dict[str, int] = {}
        self.vocab_size: int = 0
        # index 0 unused; orders 1..n
        self.counts: list[dict[str, np.ndarray[np.int32]]] = [{}] + [
            {} for _ in range(n)
        ]
        self.totals: list[dict[str, int]] = [{}] + [
            {} for _ in range(n)
        ]
        self.types: list[dict[str, int]] = [{}] + [
            {} for _ in range(n)
        ]
        self._score_cache: dict[tuple[str, str], float] = {}

    # ---- training ---------------------------------------------------------

    def train(self, path: str) -> None:
        words = self._parse_freq_list(path)
        self._build_alphabet(words)
        click.echo(
            f"  alphabet: {self.vocab_size}  |  "
            f"word types: {len(words):,}",
            err=True,
        )
        for word, freq in tqdm(
            words,
            desc="Training LM",
            unit="word",
            total=len(words),
        ):
            self._add_word(word, freq)

    @staticmethod
    def _parse_freq_list(path: str) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                v1, v2 = parts
                try:
                    freq, word = (
                        (int(v1), v2) if v1.isdigit() else (int(v2), v1)
                    )
                except ValueError:
                    continue
                out.append((word.lower(), freq))
        return out

    def _build_alphabet(self, words: list[tuple[str, int]]) -> None:
        chars: set[str] = {" "}
        for w, _ in words:
            chars.update(w)
        for i, c in enumerate(sorted(chars)):
            self.char2id[c] = i
        self.vocab_size = len(self.char2id)

    def _add_word(self, word: str, freq: int) -> None:
        pad = " " * (self.n - 1)
        padded = f"{pad}{word} "
        get_id = self.char2id.get

        for i in range(len(padded)):
            cid = get_id(padded[i])
            if cid is None:
                continue
            for order in range(1, self.n + 1):
                if i < order - 1:
                    continue
                ctx = padded[i - (order - 1) : i]
                counts_o = self.counts[order]
                totals_o = self.totals[order]
                types_o  = self.types[order]
                arr = counts_o.get(ctx)
                if arr is None:
                    arr = np.zeros(self.vocab_size, dtype=np.int32)
                    counts_o[ctx] = arr
                    totals_o[ctx] = 0
                    types_o[ctx]  = 0
                if arr[cid] == 0:
                    types_o[ctx] += 1
                arr[cid] += freq
                totals_o[ctx] += freq

    # ---- inference --------------------------------------------------------

    def get_probability(self, context: str, char: str, order: int) -> float:
        if order <= 0:
            return 1.0 / self.vocab_size if self.vocab_size else 1e-7

        n_c = self.totals[order].get(context, 0)
        r_c = self.types[order].get(context, 0)

        if n_c == 0:
            return self.get_probability(context[1:], char, order - 1)

        cid = self.char2id.get(char)
        c_i = 0
        if cid is not None:
            arr = self.counts[order].get(context)
            if arr is not None and cid < len(arr):
                c_i = int(arr[cid])

        if c_i > 0:
            return c_i / (n_c + r_c)
        else:
            return (r_c / (n_c + r_c)) * self.get_probability(
                context[1:], char, order - 1
            )

    def score_token(self, token: str) -> tuple[float, float]:
        """Return (max_surprisal, geometric_mean_probability)."""
        token = token.lower()
        pad = " " * (self.n - 1)
        padded = f"{pad}{token} "
        surprisals: list[float] = []

        for i in range(self.n - 1, len(padded)):
            ctx = padded[i - (self.n - 1) : i]
            char = padded[i]
            prob = self.get_probability(ctx, char, self.n)
            if prob <= 0:
                prob = 1e-10
            surprisals.append(-math.log2(prob))

        if not surprisals:
            return 0.0, 0.0
        return max(surprisals), 2 ** -(sum(surprisals) / len(surprisals))


    def sample_next_char(self, context: str, temperature: float = 1.0) -> str:
        """Sample a character from the distribution at the given temperature."""
        context = context[-(self.n - 1) :] if self.n > 1 else ""
        probs = np.zeros(self.vocab_size)
        
        # Invert char2id for lookup
        id2char = {i: c for c, i in self.char2id.items()}
        
        for char, cid in self.char2id.items():
            probs[cid] = self.get_probability(context, char, self.n)

        if temperature <= 0:
            return id2char[np.argmax(probs)]
        
        # Apply temperature
        probs = np.exp(np.log(probs + 1e-10) / temperature)
        probs /= probs.sum()
        
        return np.random.choice(list(id2char.values()), p=probs)

    def generate(self, length: int = 20, temperature: float = 1.0) -> str:
        """Generate a string of text."""
        res = " " * (self.n - 1)
        for _ in range(length):
            next_c = self.sample_next_char(res, temperature)
            res += next_c
            if next_c == " " and len(res.strip()) > 0: # Stop at word end or continue
                break
        return res.strip()


    def complete(self, prefix: str, max_chars: int = 20, temperature: float = 0.0) -> str:
        """Complete a word given a prefix."""
        res = prefix
        for _ in range(max_chars):
            # We use a very low temperature or 0 (greedy) for completions
            next_c = self.sample_next_char(res, temperature=temperature)
            
            # If the model predicts a space, the word is finished
            if next_c == " ":
                break
            res += next_c
        return res

    # ---- scoring helpers --------------------------------------------------

    def compute_score(self, token: str, reduction: str) -> float:
        key = (token, reduction)
        if key in self._score_cache:
            return self._score_cache[key]
        max_s, avg_p = self.score_token(token)
        result = max_s if reduction == "max" else (
            -math.log2(avg_p) if avg_p > 0 else 999.0
        )
        self._score_cache[key] = result
        return result

    # ---- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the model to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        click.echo(f"[*] Model saved to {path}", err=True)

    @classmethod
    def load(cls, path: str) -> "WittenBellCharLM":
        """Deserialise a model from a pickle file."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        click.echo(f"[*] Model loaded from {path}", err=True)
        return obj


# ----- Helpers -----


def _load_lm(train: str, n: int) -> WittenBellCharLM:
    lm = WittenBellCharLM(n=n)
    click.echo(f"[*] Training on {train} (n={n})…", err=True)
    lm.train(train)
    return lm

# ------ Helpers -----

def _load_or_prompt(wb_pkl: str | None) -> WittenBellCharLM:
    """Load model from pkl, or prompt user to build one and terminate."""
    if wb_pkl and os.path.isfile(wb_pkl):
        return WittenBellCharLM.load(wb_pkl)
    click.echo(
        "[!] No model file supplied or found.\n"
        "    Build one first:\n\n"
        "      build-wb --train <freq_list> --output <model.pkl> [--n 3]\n",
        err=True,
    )
    raise SystemExit(1)

# ------ CLI -----

@click.command("build")
@click.option("--train", required=True, help="Frequency list path")
@click.option("--output", required=True, help="Output .pkl path")
@click.option("--n", default=3, show_default=True)
def build(train, output, n):
    """Train a WittenBell char LM and save it to a pickle file."""
    lm = _load_lm(train, n)
    lm.save(output)


@click.command("interact")
@click.option("--wb-pkl", default=None, help="Path to a pre-built .pkl model")
def interact(wb_pkl):
    """Interactive playground for generation, scoring, and completion."""

    lm = _load_or_prompt(wb_pkl)
    # Store temperature in a local state for runtime config
    state = {"temp": 1.0}
    
    click.echo("\n[Modes]")
    click.echo("  g           : generate random words")
    click.echo("  s <word>    : score a specific word (shows all metrics)")
    click.echo("  c <pref>    : complete a word prefix")
    click.echo("  t <float>   : set temperature (current: {})".format(state["temp"]))
    click.echo("  q           : quit")
    
    while True:
        try:
            raw_input = input(f"\n(lm T={state['temp']:.2f}) > ").strip()
            if not raw_input: continue
            
            parts = raw_input.split(maxsplit=1)
            action = parts[0].lower()
            
            if action == 'q':
                break

            elif action == 't':
                if len(parts) < 2:
                    click.echo(f"Current temperature: {state['temp']}")
                    continue
                try:
                    state["temp"] = float(parts[1])
                    click.echo(f"Temperature set to {state['temp']}")
                except ValueError:
                    click.echo("Invalid temperature value.")

            elif action == 'g':
                click.echo(f"Generating 10 samples (T={state['temp']}):")
                for _ in range(10):
                    gen = lm.generate(length=15, temperature=state["temp"])
                    click.echo(f"  - {gen}")
            
            elif action == 's':
                if len(parts) < 2:
                    click.echo("Usage: s <token>")
                    continue
                token = parts[1]
                max_s, geom_p = lm.score_token(token)
                # Mean surprisal is -log2 of geometric mean probability
                mean_s = -math.log2(geom_p) if geom_p > 0 else 999.0
                
                click.echo(f"Token: {token}")
                click.echo(f"  Max Surprisal:     {max_s:.4f}")
                click.echo(f"  Mean Surprisal:    {mean_s:.4f}")
                click.echo(f"  Geom Mean Prob:    {geom_p:.6f}")

            elif action == 'c':
                if len(parts) < 2:
                    click.echo("Usage: c <prefix>")
                    continue
                prefix = parts[1]
                # Completion now uses the configured temperature
                completion = lm.complete(prefix, temperature=state["temp"])
                click.echo(f"Prefix:     {prefix}")
                click.echo(f"Completion: {completion}")
            
            else:
                click.echo("Unknown command. Use 'g', 's', 'c', 't', or 'q'.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            click.echo(f"Error: {e}")


@click.command("tune")
@click.option("--input", "input_path", required=True, help="Plain text file")
@click.option("--wb-pkl", default=None, help="Path to a pre-built .pkl model")
@click.option(
    "--reduction",
    type=click.Choice(["max", "mean"]),
    default="mean",
    show_default=True,
)
def tune(input_path, wb_pkl, reduction):
    """Interactive threshold tuning on a plain-text file."""
    lm = _load_or_prompt(wb_pkl)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # whitespace tokenisation – one sentence per non-blank line
    sentences: list[dict] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        scored = []
        for t in line.split():
            if not any(c.isalnum() for c in t):
                continue
            score = lm.compute_score(t, reduction)
            _, avg_p = lm.score_token(t)
            scored.append({"text": t, "score": score, "prob": avg_p})
        if scored:
            sentences.append({"text": line, "tokens": scored})

    click.echo("\n[!] Entering Interactive Mode")
    click.echo("Commands: Enter a float for threshold, or 'q' to quit.")

    while True:
        try:
            val = input(f"\nSet Threshold (reduction: {reduction}) > ")
            if val.lower() == "q":
                break
            threshold = float(val)
        except (ValueError, EOFError):
            click.echo("Invalid input. Enter a number.")
            continue

        for sent in sentences:
            suspects = [
                f"{t['text']}(S:{t['score']:.1f}, P:{t['prob']:.4f})"
                for t in sent["tokens"]
                if t["score"] > threshold
            ]
            all_words = [
                f"{t['text']}(S:{t['score']:.1f}, P:{t['prob']:.4f})"
                for t in sent["tokens"]
            ]
            status = "[!] OOD" if suspects else "[ ] CLN"
            click.echo(f"\n{status}: {sent['text']}")
            click.echo(f"    All tokens: {', '.join(all_words)}")
            if suspects:
                click.echo(f"    Suspects:   {', '.join(suspects)}")


@click.group("wb")
def wb():
    """Witten Bell language model."""


wb.add_command(build)
wb.add_command(tune)
wb.add_command(interact)


if __name__ == "__main__":
    wb()
