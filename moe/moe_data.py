"""
moe_data.py - prepare training data for the small MoE language model.
Streams FineWeb-Edu (sample-10BT), trains a byte-level BPE tokenizer on
the first --tok_docs documents, and writes uint16 token files:
  tokenizer.json, val.bin (fresh documents), train.bin
to --local, then copies them to --out (Drive) for reuse.
"""
import argparse
import itertools
import os
import shutil
import time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

T0 = time.time()


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/drive/MyDrive/moe/data")
    ap.add_argument("--local", default="/content/moe_data")
    ap.add_argument("--tokens", type=int, default=300_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--tok_docs", type=int, default=50_000)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.local, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    it = (r["text"] for r in ds)
    first = list(itertools.islice(it, a.tok_docs))
    log("downloaded %d documents for the tokenizer" % len(first))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab, special_tokens=["<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False)
    tok.train_from_iterator(first, trainer=trainer)
    tok.save(os.path.join(a.local, "tokenizer.json"))
    eos = tok.token_to_id("<|eos|>")
    log("tokenizer trained: vocab %d" % tok.get_vocab_size())

    def write(path, source, limit):
        n, last, buf = 0, 0, []
        with open(path, "wb") as f:
            for text in source:
                buf.append(text)
                if len(buf) < a.batch:
                    continue
                ids = []
                for e in tok.encode_batch(buf):
                    ids.extend(e.ids)
                    ids.append(eos)
                np.asarray(ids, dtype=np.uint16).tofile(f)
                n += len(ids)
                buf = []
                if n - last >= 10_000_000:
                    last = n
                    log("  %s: %.0fM tokens" % (os.path.basename(path),
                                                 n / 1e6))
                if n >= limit:
                    break
        return n

    nv = write(os.path.join(a.local, "val.bin"), it, a.val_tokens)
    log("val.bin: %d tokens" % nv)
    nt = write(os.path.join(a.local, "train.bin"),
               itertools.chain(first, it), a.tokens)
    log("train.bin: %d tokens" % nt)
    os.makedirs(a.out, exist_ok=True)
    for f in ("tokenizer.json", "val.bin", "train.bin"):
        shutil.copy(os.path.join(a.local, f), os.path.join(a.out, f))
    log("copied to %s" % a.out)


if __name__ == "__main__":
    main()
