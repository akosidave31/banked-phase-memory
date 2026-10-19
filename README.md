# Banked phase memory: two-hop composition on FB15k-237

A memory made of random, never-trained phase vectors that answers zero-shot
two-hop questions over the FB15k-237 knowledge graph (272,115 facts).

## Result

5 seeds, 20 relation chains (4,571 questions), 1,024 complex numbers per
vector, top-5 beam:

| Memory | Numbers stored | Hop-2 fact retrieval | Two-hop (graph-correct) | Two-hop (filtered) |
|---|---|---|---|---|
| One superposed vector | 1,024 | 0.001 | 0.001 | 0.001 |
| Banks, ~64 facts each | 4.4M | 0.962 | 0.903 | 0.705 |
| Banks, ~16 facts each | 17.4M | 0.995 | **0.980 ± 0.002** | 0.774 |
| Banks, ~4 facts each | 69.7M | 0.999 | 0.994 | 0.797 |

Chance is about 0.0013 (graph-correct) and 0.0011 (filtered).
With 512 numbers per vector and ~16 facts per bank: 0.938 ± 0.005.

Pre-registered bar, set before the run (banks ~16 facts, beam 5,
graph-correct >= 0.90, spread across seeds <= 0.02): **passed**.

## Why this matters

Kumar (2026) showed that holographic memory with one superposed vector fails
at zero-shot two-hop questions on FB15k-237, because second-hop facts cannot
be retrieved under superposition. The paper lists a memory with explicitly
allocated capacity as untested. This repo tests that idea.

## How it works

- Every training triple (h, r, t) is written by algebra:
  `fact = E_h * R_r * shift(E_t)`. E and R are random unit phasors; `shift`
  is a cyclic shift that gives the binding a direction. Nothing is trained.
- Facts with the same (h, r) go to the less-loaded of 2 banks. The banks are
  picked from sign bits of random projections of the key `E_h * R_r`, so no
  IDs are used.
- A lookup reads 2 banks and keeps the more confident read.
- Two-hop: take the top-5 intermediates from hop 1, read hop 2 for each, and
  keep the path with the highest product of the two scores.

## Run it

On Google Colab with a GPU (torch is preinstalled):

    python freqmem_kg11.py            # 512 numbers per vector
    python freqmem_kg11.py --D 1024   # 1,024 numbers per vector

The script downloads FB15k-237 `train.txt` automatically. Each run takes
about 2-4 minutes on a T4 GPU.

## Limits

- Storage is about 64 complex numbers per fact at ~16 facts per bank, far
  larger than a triple table. The point is an editable vector memory, not
  compression.
- The final clean-up compares against all 14,505 entities.
- One dataset. Chains are selected by support, answers come from the
  training graph, and atomic scores are retrieval of stored facts rather
  than link prediction. This differs from the paper's protocol.
- The filtered metric removes answers directly connected to the subject.
  For attribute chains this rejects natural answers: a perfect lookup's most
  common answer survives the filter only 63.5% of the time.

## References

- R. Kumar, "Holographic Memory for Zero-Shot Compositional Reasoning in
  Knowledge Graphs: A Mechanistic Study of Where and Why It Fails" (2026),
  arXiv:2606.24948.
- FB15k-237: Toutanova et al., 2015.

## License

MIT
