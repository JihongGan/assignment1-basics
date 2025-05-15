from gpt.tokenizer import train, EOT
import json
import base64
from gpt.config import DATA_DIR


def train_tinystories():
    vocab, merges = train(
        DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
        10000,
        [EOT],
    )

    out = DATA_DIR / "tinystories_bpe"
    out.mkdir(exist_ok=True)

    # ── vocab: id → bytes (stored as base64)
    with open(out / "vocab.json", "w") as f:
        json.dump(
            {
                str(i): base64.b64encode(tok).decode("ascii")
                for i, tok in vocab.items()
            },
            f,
        )

    # ── merges: one "token1 token2" per line, as base64
    with open(out / "merges.txt", "w") as f:
        for a, b in merges:
            a_str = base64.b64encode(a).decode("ascii")
            b_str = base64.b64encode(b).decode("ascii")
            f.write(f"{a_str} {b_str}\n")


if __name__ == "__main__":
    train_tinystories()
