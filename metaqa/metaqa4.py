"""
metaqa4.py - MetaQA edit test: change facts in the memory and check that
answers follow at once, with no retraining of anything.
1. Build the memory and train the path parser (as metaqa3, one seed).
2. Answer every test question with the unedited memory.
3. Pick N facts (director / writer / actor / year / genre / language of
   films that test questions pass through) and change each object to
   another real entity of the same relation. Each change is applied in
   place: subtract the old fact vector, add the new one, both directions.
4. Gold = exact lookup on the CURRENT knowledge base along each question's
   path (before and after the edits).
5. Answer every question again with the edited memory.
Reports: affected / unaffected accuracy, old-answer leaks, a control (old
memory on the edited questions), and the time per edit.
Needs metaqa1.py, metaqa2.py, metaqa3.py and the MetaQA files.
"""
import argparse
import collections
import math
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import metaqa1 as M
import metaqa2 as M2
import metaqa3 as M3

T0 = time.time()
DEV = M.DEV
EDIT_RELS = ["directed_by", "written_by", "starred_actors", "release_year",
             "has_genre", "in_language"]


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


class EditMem(M2.SpreadMem):
    """Spread memory (one bank per key slot) that remembers where every
    fact lives, so single facts can be changed in place."""

    @torch.no_grad()
    def build(self, S, R_, O, L, n_ent, n_rel, reserve=256, chunk=32768):
        n = len(S)
        cnt = collections.Counter()
        slot = []
        self.fslot = {}
        for x, rr, y in zip(S, R_, O):
            j = cnt[(x, rr)] // L
            slot.append(j)
            self.fslot[(x, rr, y)] = j
            cnt[(x, rr)] += 1
        deg = torch.zeros(n_ent, n_rel, dtype=torch.long)
        for (x, rr), c in cnt.items():
            deg[x, rr] = c
        self.deg, self.L = deg.to(DEV), L
        g = torch.Generator().manual_seed(12345)
        u = torch.rand(max(slot) + 1 + reserve, self.D, generator=g) * 2 - 1
        self.Sv = torch.exp(1j * math.pi * u).to(DEV)
        self.nb = nb = max(n // L, 1)
        st, rt, ot, jt = (torch.tensor(S), torch.tensor(R_), torch.tensor(O),
                          torch.tensor(slot))

        def key(i):
            return (self.addr[st[i:i + chunk].to(DEV)]
                    * self.R[rt[i:i + chunk].to(DEV)]
                    * self.Sv[jt[i:i + chunk].to(DEV)])
        b1s, b2s = [], []
        for i in range(0, n, chunk):
            x1, x2 = self.route(key(i), nb)
            b1s.append(x1.cpu())
            b2s.append(x2.cpu())
        self.load, self.unit, fb = [0] * nb, {}, []
        for x, rr, j, x1, x2 in zip(S, R_, slot, torch.cat(b1s).tolist(),
                                    torch.cat(b2s).tolist()):
            b = self.unit.get((x, rr, j))
            if b is None:
                b = x2 if self.load[x2] < self.load[x1] else x1
                self.unit[(x, rr, j)] = b
            self.load[b] += 1
            fb.append(b)
        fb = torch.tensor(fb)
        re_ = torch.zeros(nb, self.D, device=DEV)
        im = torch.zeros(nb, self.D, device=DEV)
        for i in range(0, n, chunk):
            fact = key(i) * torch.roll(self.addr[ot[i:i + chunk].to(DEV)],
                                       1, -1)
            ix = fb[i:i + chunk].to(DEV)
            re_.index_add_(0, ix, fact.real.contiguous())
            im.index_add_(0, ix, fact.imag.contiguous())
        self.B = torch.complex(re_, im)

    def _vec(self, s, r, j, o):
        return (self.addr[s] * self.R[r] * self.Sv[j]
                * torch.roll(self.addr[o], 1, -1))

    @torch.no_grad()
    def remove(self, s, r, o):
        j = self.fslot.pop((s, r, o))
        self.B[self.unit[(s, r, j)]] -= self._vec(s, r, j, o)

    @torch.no_grad()
    def add(self, s, r, o):
        j = int(self.deg[s, r]) // self.L
        self.deg[s, r] += 1
        b = self.unit.get((s, r, j))
        if b is None:
            k = (self.addr[s] * self.R[r] * self.Sv[j])[None]
            x1, x2 = self.route(k, self.nb)
            x1, x2 = int(x1), int(x2)
            b = x2 if self.load[x2] < self.load[x1] else x1
            self.unit[(s, r, j)] = b
        self.load[b] += 1
        self.fslot[(s, r, o)] = j
        self.B[b] += self._vec(s, r, j, o)

    @torch.no_grad()
    def change(self, s, r, o_old, o_new, NB):
        j = self.fslot.pop((s, r, o_old))
        self.B[self.unit[(s, r, j)]] += (self._vec(s, r, j, o_new)
                                        - self._vec(s, r, j, o_old))
        self.fslot[(s, r, o_new)] = j
        self.remove(o_old, r + NB, s)
        self.add(o_new, r + NB, s)


def run_mem(mem, Q, a):
    groups = collections.defaultdict(list)
    for i, q in enumerate(Q):
        groups[(q["pp"], q["pm"])].append(i)
    best = [None] * len(Q)
    bs = max(8, 2560 // a.K)
    for (path, mode), ids in groups.items():
        for k in range(0, len(ids), bs):
            ch = ids[k:k + bs]
            top = torch.tensor([Q[i]["tid"] for i in ch], device=DEV)
            res = M2.answer(mem, top, list(path), a.K, mode, True,
                            a.cap).tolist()
            for i, b in zip(ch, res):
                best[i] = b
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/mqa")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--edits", type=int, default=1000)
    ap.add_argument("--edit_seed", type=int, default=7)
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--k_layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sample2", type=int, default=150)
    ap.add_argument("--sample3", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names, idx, rels, trip, bad = M.load_kb(a.root)
    NB = len(rels)
    NREL = 2 * NB
    low = {}
    for i, nm in enumerate(names):
        low.setdefault(nm.lower(), i)

    def look(x):
        i = idx.get(x)
        return i if i is not None else low.get(x.lower())

    adj, outr = collections.defaultdict(list), collections.defaultdict(set)
    S, Rr, O = [], [], []
    for s, r, o in trip:
        for x, rr, y in ((s, r, o), (o, r + NB, s)):
            adj[(x, rr)].append(y)
            outr[x].add(rr)
            S.append(x)
            Rr.append(rr)
            O.append(y)
    is_movie = {e for e, rs in outr.items() if any(r < NB for r in rs)}
    data = {}
    for hop in (1, 2, 3):
        tr = M3.load_split(a.root, hop, "train", look)
        te = M3.load_split(a.root, hop, "test", look)
        if tr and te:
            data[hop] = (tr, te)
    hops = sorted(data)

    tpath = {}
    for hop in hops:
        groups = collections.defaultdict(list)
        for q, tid, gold, t in data[hop][0]:
            if tid is not None:
                groups[t].append((tid, gold))
        ns = a.sample2 if hop <= 2 else a.sample3
        for t, items in groups.items():
            sc = M2.recover2(adj, outr, items[:ns], hop)
            if sc:
                tpath[(hop, t)] = max(sc, key=sc.get)
        log("%d-hop paths recovered" % hop)
    train = []
    for hop in hops:
        for q, tid, gold, t in data[hop][0]:
            lab = tpath.get((hop, t))
            if lab and lab[0]:
                train.append((q, list(lab[0]), int(lab[1])))

    mem = EditMem(len(names), NREL, a.D, a.seed)
    sync()
    t0 = time.time()
    mem.build(S, Rr, O, a.L, len(names), NREL)
    sync()
    t_build = time.time() - t0
    log("memory built in %.1fs: %d banks" % (t_build, mem.nb))

    dtype = torch.float16 if DEV == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.lm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.lm, dtype=dtype)
    lm = lm.to(DEV).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    torch.manual_seed(a.seed)
    m = M3.PathParser(lm.config.hidden_size, NREL).to(DEV)
    log("training parser")
    M3.train_parser(m, lm, tok, train, a, random.Random(a.seed + 1), NREL)
    Q = []
    for hop in hops:
        te = data[hop][1]
        preds = M3.parse(m, lm, tok, te, a, is_movie, NB)
        for x, (path, mode) in zip(te, preds):
            lab = tpath.get((hop, x[3]))
            if x[1] is None or not lab or not lab[0]:
                continue
            Q.append({"hop": hop, "tid": x[1], "pp": path, "pm": mode,
                      "lp": lab[0], "lm": int(lab[1])})
    del m, lm
    if DEV == "cuda":
        torch.cuda.empty_cache()
    log("%d test questions parsed" % len(Q))

    gold_old = [M2.follow2(adj, q["tid"], q["lp"], q["lm"]) for q in Q]
    before = run_mem(mem, Q, a)
    log("answered with the unedited memory")

    erel = {i for i, x in enumerate(rels) if x in EDIT_RELS}
    touched = set()
    for q in Q:
        cur = {q["tid"]}
        if q["tid"] in is_movie:
            touched.add(q["tid"])
        for r in q["lp"]:
            nxt = set()
            for e in cur:
                nxt.update(adj.get((e, r), ()))
            cur = nxt
            if len(cur) > 200:
                break
            touched.update(e for e in cur if e in is_movie)
    facts = set(trip)
    cand = [f for f in trip if f[1] in erel and f[0] in touched]
    rng = random.Random(a.edit_seed)
    chosen = rng.sample(cand, min(a.edits, len(cand)))
    objs = {r: sorted({o for s, rr, o in trip if rr == r}) for r in erel}
    edits = []
    for s, r, o in chosen:
        for _ in range(50):
            o2 = rng.choice(objs[r])
            if o2 != o and (s, r, o2) not in facts:
                break
        else:
            continue
        facts.discard((s, r, o))
        facts.add((s, r, o2))
        edits.append((s, r, o, o2))
    for s, r, o, o2 in edits:
        adj[(s, r)].remove(o)
        adj[(s, r)].append(o2)
        adj[(o, r + NB)].remove(s)
        adj[(o2, r + NB)].append(s)
        outr[o2].add(r + NB)
        if not adj[(o, r + NB)]:
            outr[o].discard(r + NB)
    sync()
    t0 = time.time()
    for s, r, o, o2 in edits:
        mem.change(s, r, o, o2, NB)
    sync()
    t_edit = time.time() - t0
    log("%d edits applied to memory in %.3fs" % (len(edits), t_edit))

    gold_new = [M2.follow2(adj, q["tid"], q["lp"], q["lm"]) for q in Q]
    after = run_mem(mem, Q, a)
    log("answered with the edited memory")
    sync()
    t0 = time.time()
    for s, r, o, o2 in reversed(edits):
        mem.change(s, r, o2, o, NB)
    sync()
    t_rev = time.time() - t0
    restored = run_mem(mem, Q, a)
    log("reverted all edits in %.3fs and answered again" % t_rev)

    st = {h: collections.Counter() for h in hops}
    for i, q in enumerate(Q):
        c = st[q["hop"]]
        go, gn = gold_old[i], gold_new[i]
        if go:
            c["n_before"] += 1
            c["hit_before"] += before[i] in go
        if not gn:
            c["empty_after"] += 1
            continue
        if go != gn:
            c["n_aff"] += 1
            c["hit_aff"] += after[i] in gn
            c["ctrl"] += before[i] in gn
            c["leak"] += after[i] in (go - gn)
            if not (go & gn):
                c["n_full"] += 1
                c["hit_full"] += after[i] in gn
                c["ctrl_full"] += before[i] in gn
                c["leak_full"] += after[i] in go
        else:
            c["n_un"] += 1
            c["hit_un"] += after[i] in gn
            c["same_un"] += after[i] == before[i]

    for i, q in enumerate(Q):
        c = st[q["hop"]]
        c["n_all"] += 1
        c["same_rev"] += restored[i] == before[i]
        if gold_old[i]:
            c["n_rev"] += 1
            c["hit_rev"] += restored[i] in gold_old[i]

    def f(c, k, n):
        return "%7.4f" % (c[k] / c[n]) if c[n] else "      -"

    print()
    print("=== MetaQA edit test: %d fact changes, nothing retrained "
          "(D=%d, L=%d, K=%d) ===" % (len(edits), a.D, a.L, a.K))
    ec = collections.Counter(rels[r] for _, r, _, _ in edits)
    print("edits by relation: " + ", ".join("%s %d" % kv for kv in
                                             ec.most_common()))
    for s, r, o, o2 in edits[:3]:
        print("  e.g. %s | %s | %s  ->  %s" % (names[s][:30], rels[r],
                                              names[o][:24], names[o2][:24]))
    print("edit time: %.3f s total, %.2f ms per edit (full memory rebuild: "
          "%.1f s)" % (t_edit, 1000 * t_edit / max(len(edits), 1), t_build))
    print("revert time: %.3f s for all %d edits" % (t_rev, len(edits)))
    print("gold = exact lookup on the current knowledge base along each "
          "question's path")
    print()
    print("%-44s %s" % ("", "  ".join("%7s" % ("%d-hop" % h) for h in hops)))
    rows = [("questions", None),
            ("affected by the edits", "n_aff"),
            ("before edits: Hits@1 (all)", ("hit_before", "n_before")),
            ("after edits: Hits@1, unaffected questions",
             ("hit_un", "n_un")),
            ("after edits: Hits@1, AFFECTED questions",
             ("hit_aff", "n_aff")),
            ("control: OLD memory on affected questions", ("ctrl", "n_aff")),
            ("old answer leaked (affected)", ("leak", "n_aff")),
            ("STRICT: answers fully replaced", "n_full"),
            ("STRICT after edits: Hits@1", ("hit_full", "n_full")),
            ("STRICT control: OLD memory", ("ctrl_full", "n_full")),
            ("STRICT old answer given", ("leak_full", "n_full")),
            ("after REVERT: Hits@1 vs original gold", ("hit_rev", "n_rev")),
            ("after REVERT: same answer as before edits",
             ("same_rev", "n_all")),
            ("unaffected answers unchanged", ("same_un", "n_un")),
            ("questions with no answer after edits", "empty_after")]
    for label, spec in rows:
        vals = []
        for h in hops:
            c = st[h]
            if spec is None:
                vals.append("%7d" % sum(1 for q in Q if q["hop"] == h))
            elif isinstance(spec, str):
                vals.append("%7d" % c[spec])
            else:
                vals.append(f(c, spec[0], spec[1]))
        print("%-44s %s" % (label, "  ".join(vals)))
    log("done")


if __name__ == "__main__":
    main()
