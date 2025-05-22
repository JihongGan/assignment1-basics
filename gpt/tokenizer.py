import multiprocessing as mp
import os
from collections import Counter, defaultdict
from typing import Self, Iterable, Iterator
import regex as re
import json
import base64

PAT = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        # longest tokens first so overlapping tokens keep the longest intact
        self.special_tokens = (
            sorted(special_tokens, key=len, reverse=True) if special_tokens else None
        )

        self.id_of: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.rank: dict[tuple[bytes, bytes], int] = {p: i for i, p in enumerate(merges)}

    @classmethod
    def from_training_data(
        cls, vocab_size: int, special_tokens: list[str], input_path: str
    ) -> Self:
        vocab, merges = train(input_path, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        # Load vocab: JSON mapping from str(id) → base64-encoded bytes
        with open(vocab_filepath, "r") as f:
            vocab_json = json.load(f)
            vocab = {
                int(token_id): base64.b64decode(token_b64)
                for token_id, token_b64 in vocab_json.items()
            }

        # Load merges: each line is "base64(a) base64(b)"
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r") as f:
            for line in f:
                a_b64, b_b64 = line.strip().split()
                merges.append((base64.b64decode(a_b64), base64.b64decode(b_b64)))
        return cls(vocab, merges, special_tokens)

    def _bpe(self, word: bytes) -> list[bytes]:
        symbols = [bytes([b]) for b in word]
        while True:
            best = None
            best_rank = 1e12
            for a, b in zip(symbols, symbols[1:]):
                r = self.rank.get((a, b))
                if r is not None and r < best_rank:
                    best_rank = r
                    best = (a, b)
            if best is None:
                break
            merged = best[0] + best[1]
            out: list[bytes] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self.special_tokens:
            pat = re.compile(f"({'|'.join(map(re.escape, self.special_tokens))})")
            parts = pat.split(text)
        else:
            parts = [text]
        for part in parts:
            if not part:
                continue
            if self.special_tokens and part in self.special_tokens:
                tok_id = self.id_of[part.encode()]
                ids.append(tok_id)
                continue
            for m in PAT.finditer(part):
                for sym in self._bpe(m.group().encode()):
                    ids.append(self.id_of[sym])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files
        that we cannot directly load into memory.
        """
        for chunk in iterable:
            for tid in self.encode(chunk):
                yield tid

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.vocab[t] for t in tokens).decode("utf-8", errors="ignore")


def find_chunk_boundaries(
    input_path: str, desired_chunks: int, split_tokens: list[bytes]
) -> list[int]:
    """Split input_path into ~desired_chunks pieces aligned on any of the split_tokens boundaries."""
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        chunk = size // desired_chunks
        bounds = [i * chunk for i in range(desired_chunks + 1)]
        bounds[-1] = size
        step = 4096  # 4 KB
        for i in range(1, len(bounds) - 1):
            pos = bounds[i]
            f.seek(pos)
            while True:
                buf = f.read(step)
                if not buf:
                    bounds[i] = size
                    break
                # Find the earliest occurrence of any split token
                positions = [buf.find(token) for token in split_tokens]
                valid_positions = [p for p in positions if p >= 0]
                if valid_positions:
                    j = min(valid_positions)
                    bounds[i] = pos + j
                    break
                pos += step
    return sorted(set(bounds))


def pretokenize_str(text: str) -> Counter[bytes]:
    ctr = Counter()
    for m in PAT.finditer(text):
        ctr[m.group().encode()] += 1
    return ctr


def pretokenize_chunk(
    chunk: str, special_tokens: list[str] | None = None
) -> Counter[bytes]:
    if not special_tokens:
        return pretokenize_str(chunk)
    special_tokens_pat = re.compile("|".join(map(re.escape, special_tokens)))
    ctr = Counter()
    for part in special_tokens_pat.split(chunk):
        if part:
            ctr.update(pretokenize_str(part))
    return ctr


def pretokenize_file(
    filepath: str,
    special_tokens: list[str] | None = None,
    workers: int = mp.cpu_count(),
) -> Counter[bytes]:
    bounds = find_chunk_boundaries(
        filepath, workers * 2, [tok.encode() for tok in special_tokens]
    )
    chunks: list[str] = []
    with open(filepath, "rb") as f:
        for s, e in zip(bounds[:-1], bounds[1:]):
            f.seek(s)
            chunks.append(f.read(e - s).decode("utf-8", errors="ignore"))
    with mp.Pool(workers) as pool:
        parts = pool.starmap(
            pretokenize_chunk, [(chunk, special_tokens) for chunk in chunks]
        )
    total = Counter()
    for part in parts:
        total.update(part)
    return total


def train(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # 1. seed vocab (specials then single‑byte symbols)
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode()
    for i in range(256):
        b = bytes([i])
        if b not in vocab.values():
            vocab[len(vocab)] = b

    # 2. corpus stats
    token_freq = pretokenize_file(input_path, special_tokens)
    symbols: dict[bytes, list[bytes]] = {t: [bytes([b]) for b in t] for t in token_freq}

    # 3. build pair counts & inverted index
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    occurs: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)
    for tok, syms in symbols.items():
        freq = token_freq[tok]
        for a, b in zip(syms, syms[1:]):
            pair = (a, b)
            pair_counts[pair] += freq
            occurs[pair].add(tok)

    # 4. buckets: count → set of pairs with that count
    count_buckets: dict[int, set[tuple[bytes, bytes]]] = defaultdict(set)
    for p, c in pair_counts.items():
        if c > 0:
            count_buckets[c].add(p)
    if not count_buckets:
        return
    max_count = max(count_buckets)

    # 5. iterate merges
    while len(vocab) < vocab_size and max_count > 0:
        # pick lexicographically greatest pair among those with *max_count*
        pair = max(count_buckets[max_count])

        # remove the pair from active structures so it can't be chosen again
        count_buckets[max_count].remove(pair)
        pair_counts[pair] = 0  # zero‑out its count
        if not count_buckets[max_count]:
            # this bucket is empty; we'll drop it later after updates
            pass

        # record merge
        merged_sym = pair[0] + pair[1]
        merges.append(pair)
        vocab[len(vocab)] = merged_sym

        # tokens affected by this merge
        affected = occurs.pop(pair, set())
        for tok in affected:
            syms = symbols[tok]
            freq = token_freq[tok]
            new_syms: list[bytes] = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and (syms[i], syms[i + 1]) == pair:
                    prev_sym = new_syms[-1] if new_syms else None
                    next_sym = syms[i + 2] if i + 2 < len(syms) else None

                    # helper to decrease counts
                    def _dec(p):
                        pc = pair_counts[p]
                        if pc == 0:
                            return
                        count_buckets[pc].discard(p)
                        pair_counts[p] -= freq
                        new_c = pair_counts[p]
                        if new_c > 0:
                            count_buckets[new_c].add(p)
                        else:
                            occurs[p].discard(tok)

                    if prev_sym is not None:
                        _dec((prev_sym, syms[i]))
                    if next_sym is not None:
                        _dec((syms[i + 1], next_sym))

                    # add merged symbol & increment counts for new neighbours
                    new_syms.append(merged_sym)

                    def _inc(p):
                        pc = pair_counts[p]
                        if pc > 0:
                            count_buckets[pc].discard(p)
                        pair_counts[p] += freq
                        count_buckets[pair_counts[p]].add(p)
                        occurs[p].add(tok)

                    if prev_sym is not None:
                        _inc((prev_sym, merged_sym))
                    if next_sym is not None:
                        _inc((merged_sym, next_sym))
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            symbols[tok] = new_syms

        # find next max_count (loop until we hit a non‑empty bucket)
        while max_count > 0 and not count_buckets[max_count]:
            max_count -= 1

    return vocab, merges
