"""
pmlm_b5.py - Track B: two targeted fixes, measured separately.
  data : v1 + relations_extra.json + relations_extra2.json (new:
         located_in phrases with no place word, to break the
         "city word -> located_in, bare 'the X of' -> ceo" shortcut)
  both : same data, plus hop count read from the 4 slots (each slot
         predicts whether it is used) instead of the mean of all tokens
Data fix   = data vs last run's "new" arm (same seeds, deterministic).
Hop fix    = both vs data.
Also reports hop-count errors per held-out phrase (multi-hop only).
Needs pmlm_b.py, pmlm_b2.py, pmlm_b4.py, relations_v1.json,
relations_extra.json, relations_extra2.json.
"""
import argparse
import collections
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pmlm_b as B
import pmlm_b2 as B2
import pmlm_b4 as B4

T0 = time.time()
DEV = B.DEV
NR, HMAX = B.NR, B.HMAX
REL = B2.REL_NAMES
CONDS = B4.CONDS
CAUSES = B4.CAUSES
ARMS = {"v1": ("v1", False), "new": ("new", False),
        "data": ("data", False), "both": ("data", True)}
REF = ("reference, last run arm new (same seeds): B 0.933  0.952  "
       "0.882  0.820; hop-count errors at 4 hops 0.104")


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


class SlotHopParser(B.Parser):
    """Hop count from the slots: slot j predicts 'used'; a question
    with k hops uses slots 0..k-1. Returns hop logits like B.Parser."""

    def __init__(self, hid, maxlen=96):
        super().__init__(hid, maxlen=maxlen)
        del self.hops
        self.act = nn.Linear(self.rel.in_features, 1)
        ids = torch.arange(HMAX)
        self.register_buffer("used",
                             (ids[None, :] <= ids[:, None]).float())

    def forward(self, f, mask):
        b, T, _ = f.shape
        h = self.proj(f) + self.pos.weight[None, :T]
        h = self.enc(h, src_key_padding_mask=~mask)
        st = self.start(h).squeeze(-1).masked_fill(~mask, -1e4)
        en = self.end(h).squeeze(-1).masked_fill(~mask, -1e4)
        q = self.slots[None].expand(b, -1, -1)
        o, _ = self.att(q, h, h, key_padding_mask=~mask)
        a = self.act(o).squeeze(-1)
        hp = (F.logsigmoid(a) @ self.used.T
              + F.logsigmoid(-a) @ (1 - self.used).T)
        return st, en, self.rel(o), hp


def tally(recs, cause, hopm, phr):
    for x in recs:
        q, h = x["q"], x["h"]
        cause[h - 1][x["cause"]] += 1
        hopm[h - 1][x["ph"] - 1] += 1
        for s0, s1, hop in q["segs"]:
            r = q["rels"][hop - 1]
            st = phr[(r, q["text"][s0:s1])]
            st[0] += 1
            p = x["rr"][hop - 1]
            if p == r:
                st[1] += 1
            else:
                st[2][p] += 1
            if h >= 2:
                st[3] += 1
                st[4] += int(x["ph"] != h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="relations_v1.json")
    ap.add_argument("--extra", default="relations_extra.json")
    ap.add_argument("--extra2", default="relations_extra2.json")
    ap.add_argument("--arms", default="data,both")
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
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    arms = a.arms.split(",")
    tr, te, held = B2.load_split(a.data)
    p1, add1, drop1 = B4.add_extra(tr, held, a.data, a.extra)
    p2, add2, drop2 = B4.add_extra(p1, held, a.data, a.extra2)
    pools = {"v1": tr, "new": p1, "data": p2}
    log("extra: +%d/+%d, %d dropped; extra2: +%d/+%d, %d dropped"
        % (sum(add1["np"]), sum(add1["q"]), len(drop1),
           sum(add2["np"]), sum(add2["q"]), len(drop2)))
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
    qsets = {}
    for c, _, ci in CONDS:
        pool = tr if c == "A" else te
        opens = B.TE_OPEN if c == "D" else B.TR_OPEN
        qsets[c] = [B2.qs_nested(W, 50 + 10 * ci + h, h, pool, opens,
                                 a.nq) for h in range(1, 5)]
    acc = {arm: {c: [] for c, _, _ in CONDS} for arm in arms}
    cause = {arm: [collections.Counter() for _ in range(HMAX)]
             for arm in arms}
    hopm = {arm: [[0] * HMAX for _ in range(HMAX)] for arm in arms}
    phr = {arm: collections.defaultdict(
        lambda: [0, 0, collections.Counter(), 0, 0]) for arm in arms}
    for seed in range(a.seeds):
        for arm in arms:
            pname, slot_hops = ARMS[arm]
            cls = SlotHopParser if slot_hops else B.Parser
            torch.manual_seed(seed)
            m = cls(hid, maxlen=96).to(DEV)
            log("seed %d arm %s: training (%s)"
                % (seed, arm, "slot hops" if slot_hops else "mean hops"))
            B2.train("slots", m, lm, tok, a, random.Random(seed + 1),
                     pools[pname])
            for c, _, _ in CONDS:
                per_h = []
                for h in range(HMAX):
                    recs = B4.run_eval(m, lm, tok, mem, addr, W,
                                       qsets[c][h], a)
                    per_h.append(sum(x["cause"] == "correct"
                                     for x in recs) / len(recs))
                    if c == "B":
                        tally(recs, cause[arm], hopm[arm], phr[arm])
                acc[arm][c].append(per_h)
            log("seed %d arm %s: condition B %s"
                % (seed, arm, B2.fmt(acc[arm]["B"][-1])))
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()

    def mean(v):
        return sum(v) / len(v)

    print()
    print("=== Track B v5: data fix vs hop fix (%s, %d seeds per arm) ==="
          % (a.lm, a.seeds))
    print("extra2 adds (noun phrases/questions): " + ", ".join(
        "%s +%d/+%d" % (REL[r], add2["np"][r], add2["q"][r])
        for r in range(NR)))
    print("dropped by stem guard: %d" % (len(drop1) + len(drop2)))
    for d in drop1 + drop2:
        print("  " + d)
    print()
    print("ANSWERS (mean over seeds)")
    print("%-35s%s" % ("arm  condition",
                       "1-hop  2-hop  3-hop  4-hop   4-hop by seed"))
    for arm in arms:
        for c, desc, _ in CONDS:
            rows = acc[arm][c]
            hm = [mean([r[h] for r in rows]) for h in range(HMAX)]
            print("%-4s %s %-27s %s   %s"
                  % (arm, c, desc, B2.fmt(hm),
                     " ".join("%.3f" % r[3] for r in rows)))
    print(REF)
    print()
    print("WHY CONDITION B FAILS (fraction of questions, seeds pooled)")
    print("arm  hops  " + "  ".join("%9s" % c for c in CAUSES))
    for arm in arms:
        for h in range(HMAX):
            cnt = cause[arm][h]
            tot = max(sum(cnt.values()), 1)
            print("%-4s  %d    " % (arm, h + 1) + "  ".join(
                "%9.3f" % (cnt[c] / tot) for c in CAUSES))
    print()
    print("HOP COUNT, condition B (row = true hops, columns = "
          "predicted 1-4)")
    for arm in arms:
        for h in range(HMAX):
            row = hopm[arm][h]
            tot = max(sum(row), 1)
            print("%-4s true %d:  %s" % (arm, h + 1, "  ".join(
                "%5.3f" % (v / tot) for v in row)))
        multi = [hopm[arm][h] for h in range(1, HMAX)]
        wrong = sum(sum(r) - r[h + 1] for h, r in enumerate(multi))
        tot = max(sum(sum(r) for r in multi), 1)
        print("%-4s all multi-hop: hop-count error %.3f"
              % (arm, wrong / tot))
    print()
    print("PER HELD-OUT PHRASE, condition B (seeds pooled; hop-err = "
          "multi-hop questions with this phrase whose hop count is wrong)")
    names = " ".join("%5s" % arm for arm in arms)
    print("%-10s %-35s %4s  rel:%s  hop-err:%s  wrong guess"
          % ("relation", "phrase", "n", names, names))
    for r, fam, t in B4.held_list(a.data, te):
        st = [phr[arm].get((r, t)) for arm in arms]
        n = st[0][0] if st[0] else 0
        rel = " ".join(("%5.3f" % (s[1] / s[0])) if s and s[0]
                       else "    -" for s in st)
        hop = " ".join(("%5.3f" % (s[4] / s[3])) if s and s[3]
                       else "    -" for s in st)
        wrong = " / ".join(REL[s[2].most_common(1)[0][0]]
                           if s and s[2] else "-" for s in st)
        print("%-10s %-35s %4d      %s          %s  %s"
              % (REL[r], t, n, rel, hop, wrong))
    log("done")


if __name__ == "__main__":
    main()
