# Banked phase memory

A knowledge memory made of random, never-trained phase vectors. Facts are
written in by algebra, chained to answer multi-hop questions, and can be
changed one at a time in under a millisecond, with nothing retrained.

## Results at a glance

| Test | Result |
|---|---|
| MetaQA, full system (question in, answer out), Hits@1 | **2-hop 0.984, 3-hop 0.993** |
| MetaQA, 5,000 facts changed in place, questions whose answers were fully replaced | **0.996 / 0.993** (old memory: 0.000) |
| Time per fact edit | **0.41 ms** |
| FB15k-237, zero-shot two-hop (graph-correct) | **0.980 ± 0.002** (5 seeds) |

The only trained part is a small question parser (about 2M parameters) on
top of a frozen Qwen2.5-0.5B. The memory itself is never trained.

## MetaQA: real multi-hop questions over a movie knowledge base

MetaQA (Zhang et al., 2018) has 134,741 facts about movies (directors,
writers, actors, year, genre, language, tags, ratings) and natural 2-hop and
3-hop questions with the topic entity in brackets, e.g.
"who acted in the films directed by [Ridley Scott]".

### Full system (`metaqa/metaqa3.py`)

1. Topic entity: the bracketed name (MetaQA's standard protocol).
2. Path parser: frozen Qwen2.5-0.5B features + a small trained parser that
   predicts the hop count, the relation at each hop and how the topic is
   excluded. A schema keeps each step valid (film to person/attribute by a
   forward relation, back to films by a reverse one).
3. Memory and reasoner: the banked phase memory, beam of 20.

Training labels come from training questions only: each training template's
relation path is recovered from its question/answer pairs. No hand labels.

| | Path accuracy | Hits@1 | Memory alone, true path |
|---|---|---|---|
| 2-hop (14,872 test questions) | 0.9993 | **0.9838** | 0.9844 |
| 3-hop (14,274 test questions) | 1.0000 | **0.9929** | 0.9928 |

Three parser seeds gave identical results. The parser is essentially perfect,
so the score equals what the memory gets with the true path.

Published results (Hits@1 %, MetaQA vanilla), from Table 2 of the TransferNet
paper:

| System | 2-hop | 3-hop |
|---|---|---|
| GraftNet | 94.8 | 77.7 |
| PullNet | 99.9 | 91.4 |
| EmbedKGQA | 98.8 | 94.8 |
| TransferNet | 100 | 100 |
| **This repo** | **98.4** | **99.3** |

These systems walk the real graph or learn embeddings for it. Here, all
267,164 stored facts (with inverses) live in random vectors that are never
trained.

### Edit test (`metaqa/metaqa4.py`)

Facts are changed in place: subtract the old fact vector, add the new one,
in both directions. Gold answers are recomputed by exact lookup on the edited
knowledge base. Nothing is retrained.

| 5,000 edits (3.7% of all facts) | 2-hop | 3-hop |
|---|---|---|
| Affected questions, Hits@1 | 0.9989 | 0.9957 |
| Answers fully replaced: edited memory | **0.9955** (449 q) | **0.9926** (408 q) |
| Answers fully replaced: old memory (control) | 0.0000 | 0.0000 |
| Old answer given | 0.0000 | 0.0000 |
| Unaffected questions, before → after | 0.9993 → 0.9995 | 0.9931 → 0.9889 |
| After reverting all edits | 0.9993 | 0.9924 |

Edit time: 0.41 ms per edit; reverting 5,000 edits takes 2.1 s. With 1,000
edits, all 144 fully replaced questions were answered correctly.

### What it took to work on real data

- **One bank per key.** Each key's facts must go to the same bank. Letting
  each fact pick the emptier of two banks split multi-object keys, and a read
  of one bank lost the rest (2-hop rose from 0.93 to 0.98 after the fix).
- **Hub spreading.** Keys with thousands of objects (every film in English)
  are spread over many banks, about 4 facts each, with slot vectors. The
  memory keeps one count per key to know how many banks to read.
- **Never revisit the topic.** In 3-hop questions ("films by the same
  director"), the start film is excluded at every hop, not only at the end
  (3-hop rose from 0.88 to 0.99).

## FB15k-237: two-hop composition (`freqmem_kg11.py`)

5 seeds, 20 relation chains (4,571 questions), 1,024 complex numbers per
vector, top-5 beam:

| Memory | Numbers stored | Hop-2 fact retrieval | Two-hop (graph-correct) | Two-hop (filtered) |
|---|---|---|---|---|
| One superposed vector | 1,024 | 0.001 | 0.001 | 0.001 |
| Banks, ~64 facts each | 4.4M | 0.962 | 0.903 | 0.705 |
| Banks, ~16 facts each | 17.4M | 0.995 | **0.980 ± 0.002** | 0.774 |
| Banks, ~4 facts each | 69.7M | 0.999 | 0.994 | 0.797 |

Chance is about 0.0013. Pre-registered bar (banks ~16 facts, beam 5,
graph-correct >= 0.90, seed spread <= 0.02): **passed**.

Kumar (2026) showed that holographic memory with one superposed vector fails
at zero-shot two-hop questions on FB15k-237 and listed explicitly allocated
capacity as untested. This tests that idea.

## How it works

- Each fact (h, r, t) is written as `E_h * R_r * S_j * shift(E_t)`. E, R and
  the slot vectors S are random unit phasors; `shift` is a cyclic shift that
  gives the binding a direction. Nothing is trained.
- Banks are chosen from sign bits of random projections of the key, with
  two-choice load balancing per key slot. No IDs are needed.
- A read unbinds the key from its banks and compares the result with all
  entity vectors. Multi-hop questions follow the path with a beam, keeping
  the paths with the highest product of scores.

## Run it

On Google Colab with a T4 GPU:

    python freqmem_kg11.py --D 1024       # FB15k-237, downloads the data

    # MetaQA: put kb.txt and 2-hop/, 3-hop/ (qa_train.txt, qa_test.txt)
    # in /content/mqa, then:
    python metaqa/metaqa2.py               # memory alone, true paths
    python metaqa/metaqa3.py               # full system, ~25 min
    python metaqa/metaqa4.py --edits 5000  # edit test, ~25 min

MetaQA's official copy is on the authors' Google Drive
(github.com/yuyuz/MetaQA). These runs used the copy of the original files in
github.com/tiantiangao7/kalm-qa (metaqa/original), which has kb.txt and the
2-hop and 3-hop train and test sets.

## Language-model experiments (`lm/`)

Frozen language models (SmolLM2-135M, Qwen2.5-0.5B) plus a small parser and
this memory, on synthetic worlds with held-out phrasings. Main findings:
Qwen2.5-0.5B names every held-out relation phrase correctly zero-shot once it
knows whether the subject is a person or a company; a schema check alone
lifts 4-hop accuracy on unseen phrasings from 0.79 to 0.83; splitting long
stacked questions into phrases is the open problem.

## Limits

- MetaQA: no 1-hop results (not in the copy used); questions come from
  templates, which makes parsing easy; one memory seed.
- For "release years of films directed by X", the best recovered path matches
  MetaQA's answers only 87% of the time even by exact lookup. TransferNet
  reaches 100%, so these answers likely follow a pattern not captured here.
- Edits leave freed slots unused, so after thousands of edits heavily edited
  keys read a few extra banks and 3-hop accuracy drifts slightly (0.993 to
  0.989 after 5,000 edits). Reusing freed slots or a full rebuild (2 s) fixes
  it.
- Storage is tens of complex numbers per fact, far more than a triple table.
  The point is an editable vector memory, not compression.
- The final clean-up compares against every entity (43,234 on MetaQA).

## References

- R. Kumar, "Holographic Memory for Zero-Shot Compositional Reasoning in
  Knowledge Graphs: A Mechanistic Study of Where and Why It Fails" (2026),
  arXiv:2606.24948.
- Y. Zhang et al., "Variational Reasoning for Question Answering with
  Knowledge Graph" (MetaQA), AAAI 2018.
- J. Shi et al., "TransferNet: An Effective and Transparent Framework for
  Multi-hop Question Answering over Relation Graph", EMNLP 2021,
  arXiv:2104.07302.
- FB15k-237: Toutanova et al., 2015.

## License

MIT
