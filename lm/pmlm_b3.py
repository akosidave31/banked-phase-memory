"""
pmlm_b3.py - Track B diagnostic: which novelty hurts, unseen relation
families or unseen question openers?
Trains the slots parser (as in pmlm_b2) and tests 5 conditions on a
new world:
  A seen families   + seen openers
  B UNSEEN families + seen openers
  C seen families   + UNSEEN openers
  D UNSEEN families + UNSEEN openers   (pmlm_b2's "UNSEEN" row)
  E UNSEEN families + no opener
Needs pmlm_b.py, pmlm_b2.py and relations_v1.json in the same folder.
"""
import argparse
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pmlm_b as B
import pmlm_b2 as B2

T0 = time.time()
DEV = B.DEV


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="relations_v1.json")
    ap.add_argument("--k_layers", type=int, default=8)
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_train", type=int, default=1024)
    ap.add_argument("--nq", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    tr, te, held = B2.load_split(a.data)
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.lm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.lm, dtype=dtype)
    lm = lm.to(DEV).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    hid = lm.config.hidden_size
    log("device=%s LM %s" % (DEV, a.lm))

    addr = B.Addr(a.D)
    mem = B.Memory(a.D, a.L, 0)
    W = B.World(a.n_train, random.Random(2000))
    B.prepare(W, addr)
    mem.write(W)
    conds = [("A seen fam   + seen openers", tr, B.TR_OPEN),
             ("B UNSEEN fam + seen openers", te, B.TR_OPEN),
             ("C seen fam   + UNSEEN openers", tr, B.TE_OPEN),
             ("D UNSEEN fam + UNSEEN openers", te, B.TE_OPEN),
             ("E UNSEEN fam + no opener", te, [""])]
    qsets = {}
    for ci, (name, pool, opens) in enumerate(conds):
        qsets[name] = [B2.qs_nested(W, 50 + 10 * ci + h, h, pool,
                                    opens, a.nq)
                       for h in range(1, 5)]
    res = {name: [] for name, _, _ in conds}
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        m = B.Parser(hid, maxlen=96).to(DEV)
        log("seed %d: training slots parser" % seed)
        B2.train("slots", m, lm, tok, a, random.Random(seed + 1), tr)
        for name, _, _ in conds:
            ans = [B2.answer_acc(B2.parse_slots, m, lm, tok, mem, addr,
                                 W, q, a) for q in qsets[name]]
            allq = [q for hs in qsets[name] for q in hs]
            dg = B2.diag(B2.parse_slots, m, lm, tok, W, allq,
                         a.k_layers)
            res[name].append((ans, dg[1], dg[2][0]))
        log("seed %d evaluated" % seed)
        del m
        torch.cuda.empty_cache()

    def mean(v):
        return sum(v) / len(v)

    print()
    print("=== Track B diagnostic: families vs openers (%s, %d seeds) ==="
          % (a.lm, a.seeds))
    print("held-out families: " + "; ".join(
        "%s: %s" % (k, ", ".join(v)) for k, v in held.items()))
    print()
    print("condition                        1-hop  2-hop  3-hop  4-hop"
          "   hops  rel-h1")
    for name, _, _ in conds:
        rows = res[name]
        hop_acc = [mean([r[0][h] for r in rows]) for h in range(4)]
        print("%-31s  %s  %5.3f  %5.3f"
              % (name, B2.fmt(hop_acc), mean([r[1] for r in rows]),
                 mean([r[2] for r in rows])))
    d4 = [r[0][3] for r in res["D UNSEEN fam + UNSEEN openers"]]
    print()
    print("seed spread, condition D 4-hop: %s"
          % ", ".join("%.3f" % v for v in d4))
    log("done")


if __name__ == "__main__":
    main()
