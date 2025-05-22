"""
profile.py – run or Scalene-profile helper functions.

Examples
--------
# plain run
uv run python profile.py train_tinystories_tok

# profile with Scalene (HTML report)
uv run scalene profile.py train_tinystories_tok

# Scalene with extra flags
uv run scalene --cpu-only --html --outfile out/train.html --- \
               profile.py train_tinystories_tok
"""

from __future__ import annotations
import argparse, inspect, sys, multiprocessing as mp
from pathlib import Path

from config import DATA_DIR, EOT
from gpt.tokenizer import train


def train_tinystories_tok() -> None:
    train(DATA_DIR / "TinyStoriesV2-GPT4-train.txt", 10_000, [EOT])


_FUNCTIONS = {n: f for n, f in globals().items() if inspect.isfunction(f)}

if sys.platform == "darwin":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run or profile a helper function."
    )
    p.add_argument("func", help=f"one of: {', '.join(sorted(_FUNCTIONS))}")
    args = p.parse_args()
    _FUNCTIONS.get(args.func, lambda: p.error("unknown func"))()


if __name__ == "__main__":
    main()
