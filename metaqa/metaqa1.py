"""
metaqa1.py - MetaQA stage 1: does the banked phase memory hold up on a
real knowledge base? Memory only: true topic entity + true relation
path, so no language model is involved.
1. Load kb.txt (movie KB) and the vanilla 1/2/3-hop questions.
2. Recover each question type's relation path from the data: try
   every path on a sample of its questions and keep the one whose
   answers best match the gold answers (exact symbolic lookups).
3. Store every fact and its inverse in the banked phase memory
   (random vectors, never trained) and answer each question by
   following the path in memory with a beam.
Metric: Hits@1 (top answer is one of the gold answers), the standard
MetaQA metric. Needs the MetaQA folder (kb.txt, 1-hop/, 2-hop/, 3-hop/).
"""
import argparse
import collections
import math
import os
import re
import time
import torch

T0 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ABBR = {"directed_by": "dir", "written_by": "wri",
        "starred_actors": "act", "release_year": "year",
        "in_language": "lang", "has_tags": "tag", "has_genre": "genre",
        "has_imdb_votes": "votes", "has_imdb_rating": "rating"}


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def load_kb(root):
    names, idx, rels, rid, trip, bad = [], {}, [], {}, set(), 0

    def eid(x):
        if x not in idx:
            idx[x] = len(names)
            names.append(x)
        return idx[x]
    for line in open(os.path.join(root, "kb.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) != 3:
            bad += 1
            continue
        s, r, o = p
        if r not in rid:
            rid[r] = len(rels)
            rels.append(r)
        trip.add((eid(s), rid[r], eid(o)))
    return names, idx, rels, sorted(trip), bad


def load_qa(root, hop, split):
    d = os.path.join(root, "%d-hop" % hop)
    path = None
    for c in (os.path.join(d, "vanilla", "qa_%s.txt" % split),
              os.path.join(d, "qa_%s.txt" % split)):
        if os.path.exists(c):
            path = c
            break
    if path is None:
        return []
    qs = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    tp = os.path.join(d, "qa_%s_qtype.txt" % split)
    if os.path.exists(tp):
        qt = [l.strip() for l in open(tp, encoding="utf-8")]
    else:
        qt = [re.sub(r"\[.+?\]", "[E]", l.partition("\t")[0])
              for l in qs]
    assert len(qs) == len(qt), (hop, len(qs), len(qt))
    out = []
    for line, t in zip(qs, qt):
        q, _, ans = line.partition("\t")
        m = re.search(r"\[(.+?)\]", q)
        out.append((m.group(1) if m else None, ans.split("|"), t))
    return out


def f1(ans, gold):
    inter = len(ans & gold)
    if not inter:
        return 0.0
    p, r = inter / len(ans), inter / len(gold)
    return 2 * p * r / (p + r)


def recover(adj, outr, smp, hops, cap=5000):
    sc = collections.Counter()
    for tid, gold in smp:
        stack = [((), {tid})]
        while stack:
            path, cur = stack.pop()
            if len(path) == hops:
                for rm in ((False, True) if hops >= 2 else (False,)):
                    ans = cur - {tid} if rm else cur
                    sc[(path, rm)] += f1(ans, gold)
                continue
            rs = set()
            for e in cur:
                rs |= outr.get(e, set())
            for r in rs:
                nxt = set()
                for e in cur:
                    nxt.update(adj.get((e, r), ()))
                    if len(nxt) > cap:
                        break
                if 0 < len(nxt) <= cap:
                    stack.append((path + (r,), nxt))
    return sc


def follow(adj, tid, path, cap=50000):
    cur = {tid}
    for r in path:
        nxt = set()
        for e in cur:
            nxt.update(adj.get((e, r), ()))
        cur = nxt
        if not cur or len(cur) > cap:
            break
    return cur


class BankMem:
    def __init__(self, n_ent, n_rel, D, seed):
        g = torch.Generator().manual_seed(seed)

        def ph(n):
            u = torch.rand(n, D, generator=g) * 2 - 1
            return torch.exp(1j * math.pi * u)
        self.addr = ph(n_ent).to(DEV)
        self.R = ph(n_rel).to(DEV)
        self.P1, self.P2 = ph(31).to(DEV), ph(31).to(DEV)
        self.pow2 = (2 ** torch.arange(31, device=DEV)).long()
        self.D = D
        self.Ac = self.addr.conj().T.contiguous()

    def route(self, key, nb):
        out = []
        for P in (self.P1, self.P2):
            bits = ((key @ P.conj().T).real > 0).long()
            out.append((bits * self.pow2).sum(-1) % nb)
        return out

    @torch.no_grad()
    def write(self, s, r, o, L, chunk=32768):
        n = len(s)
        nb = max(n // L, 1)
        b1s, b2s = [], []
        for i in range(0, n, chunk):
            key = (self.addr[s[i:i + chunk].to(DEV)]
                   * self.R[r[i:i + chunk].to(DEV)])
            x1, x2 = self.route(key, nb)
            b1s.append(x1.cpu())
            b2s.append(x2.cpu())
        load, fb = [0] * nb, []
        for x1, x2 in zip(torch.cat(b1s).tolist(), torch.cat(b2s).tolist()):
            b = x2 if load[x2] < load[x1] else x1
            load[b] += 1
            fb.append(b)
        fb = torch.tensor(fb)
        re_ = torch.zeros(nb, self.D, device=DEV)
        im = torch.zeros(nb, self.D, device=DEV)
        for i in range(0, n, chunk):
            key = (self.addr[s[i:i + chunk].to(DEV)]
                   * self.R[r[i:i + chunk].to(DEV)])
            fact = key * torch.roll(self.addr[o[i:i + chunk].to(DEV)], 1, -1)
            ix = fb[i:i + chunk].to(DEV)
            re_.index_add_(0, ix, fact.real.contiguous())
            im.index_add_(0, ix, fact.imag.contiguous())
        self.B = torch.complex(re_, im)
        self.nb, self.load = nb, load

    @torch.no_grad()
    def read(self, a, r):
        key = a * self.R[r]
        b1, b2 = self.route(key, self.nb)
        kc = key.conj()
        s1 = (torch.roll(self.B[b1] * kc, -1, -1) @ self.Ac).real
        s2 = (torch.roll(self.B[b2] * kc, -1, -1) @ self.Ac).real
        pick = (s2.amax(1) > s1.amax(1))[:, None]
        return torch.where(pick, s2, s1) / self.D


@torch.no_grad()
def mem_answer(mem, topics, path, K, remove):
    n = topics.shape[0]
    ar = torch.arange(n, device=DEV)
    ca = mem.addr[topics][:, None, :]
    conf = torch.ones(n, 1, device=DEV)
    ei = topics[:, None]
    for i, r in enumerate(path):
        C = ca.shape[1]
        S = mem.read(ca.reshape(n * C, -1), r).view(n, C, -1)
        N = S.shape[-1]
        tot = conf[..., None] * S.clamp(min=0)
        if remove and i == len(path) - 1:
            tot[ar, :, topics] = -1.0
        v, idx = tot.view(n, -1).topk(K, -1)
        ei = idx % N
        conf = v.clamp(min=0)
        ca = mem.addr[ei]
    return ei[:, 0]


def mem_hits(mem, items, path, rm, K, bs=128):
    if path is None:
        return 0
    valid = [(t, g) for t, g in items if t is not None]
    hits = 0
    for i in range(0, len(valid), bs):
        ch = valid[i:i + bs]
        top = torch.tensor([t for t, _ in ch], device=DEV)
        best = mem_answer(mem, top, list(path), K, rm).tolist()
        hits += sum(int(b in g) for b, (_, g) in zip(best, ch))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/drive/MyDrive/MetaQA")
    ap.add_argument("--split", default="test")
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--Ls", default="16,4")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names, idx, rels, trip, bad = load_kb(a.root)
    NB = len(rels)
    low = {}
    for i, nm in enumerate(names):
        low.setdefault(nm.lower(), i)

    def look(x):
        i = idx.get(x)
        return i if i is not None else low.get(x.lower())

    def rname(r):
        b = ABBR.get(rels[r % NB], rels[r % NB][:6])
        return b if r < NB else "~" + b

    adj, outr = collections.defaultdict(list), collections.defaultdict(set)
    S, Rr, O = [], [], []
    for s, r, o in trip:
        for x, rr, y in ((s, r, o), (o, r + NB, s)):
            adj[(x, rr)].append(y)
            outr[x].add(rr)
            S.append(x)
            Rr.append(rr)
            O.append(y)
    log("KB: %d entities, %d relations, %d facts (%d stored with "
        "inverses), %d bad lines" % (len(names), NB, len(trip), len(S),
                                     bad))

    data = {}
    for hop in (1, 2, 3):
        groups = collections.defaultdict(list)
        miss = 0
        for topic, ans, t in load_qa(a.root, hop, a.split):
            tid = look(topic) if topic else None
            gold = {look(x) for x in ans} - {None}
            if tid is None or not gold:
                miss += 1
                tid = None
            groups[t].append((tid, gold))
        data[hop] = (groups, miss)
        log("%d-hop: %d questions, %d types, %d unresolved"
            % (hop, sum(len(v) for v in groups.values()), len(groups),
               miss))

    paths = {}
    for hop in (1, 2, 3):
        for t, items in data[hop][0].items():
            smp = [x for x in items if x[0] is not None][:a.sample]
            sc = recover(adj, outr, smp, hop)
            if sc:
                best = max(sc, key=sc.get)
                paths[(hop, t)] = (best[0], best[1],
                                   sc[best] / max(len(smp), 1))
            else:
                paths[(hop, t)] = (None, False, 0.0)
        log("%d-hop paths recovered" % hop)

    res = {}
    for hop in (1, 2, 3):
        for t, items in data[hop][0].items():
            path, rm, _ = paths[(hop, t)]
            hit = ex = 0
            if path is not None:
                for tid, gold in items:
                    if tid is None:
                        continue
                    ans = follow(adj, tid, path)
                    if rm:
                        ans = ans - {tid}
                    hit += int(bool(ans & gold))
                    ex += int(ans == gold)
            res[(hop, t)] = {"n": len(items), "sym": hit, "exact": ex}
    log("symbolic check done")

    Ls = [int(x) for x in a.Ls.split(",")]
    kc = collections.Counter(zip(S, Rr))
    loads = {}
    St, Rt, Ot = (torch.tensor(S), torch.tensor(Rr), torch.tensor(O))
    for L in Ls:
        mem = BankMem(len(names), 2 * NB, a.D, a.seed)
        mem.write(St, Rt, Ot, L)
        ld = mem.load
        loads[L] = (max(ld), sum(x for x in ld if x > 2 * L) / len(S))
        log("memory L=%d: %d banks, fullest %d" % (L, mem.nb, max(ld)))
        for hop in (1, 2, 3):
            for t, items in data[hop][0].items():
                path, rm, _ = paths[(hop, t)]
                res[(hop, t)]["L%d" % L] = mem_hits(mem, items, path, rm,
                                                    a.K)
            log("  L=%d %d-hop done" % (L, hop))
        del mem
        if DEV == "cuda":
            torch.cuda.empty_cache()

    print()
    print("=== MetaQA stage 1: memory only, true topic + true path "
          "(D=%d, beam %d) ===" % (a.D, a.K))
    print("KB: %d entities, %d relations, %d facts; stored with inverses: "
          "%d" % (len(names), NB, len(trip), len(S)))
    print("Hits@1 = top answer is a gold answer; symbolic = exact KB "
          "lookup along the recovered path (upper bound)")
    print()
    print("SUMMARY, %s split (Hits@1)" % a.split)
    cols = ["sym"] + ["L%d" % L for L in Ls]
    print("hop       n  unresolved  " + "  ".join("%7s" % c for c in
                                                  ["symbolic"] + cols[1:]))
    for hop in (1, 2, 3):
        rows = [v for (h, _), v in res.items() if h == hop]
        n = sum(v["n"] for v in rows)
        if not n:
            continue
        print("%d   %7d  %10d  %s" % (hop, n, data[hop][1], "  ".join(
            "%7.4f" % (sum(v.get(c, 0) for v in rows) / n) for c in cols)))
    print()
    print("PER QUESTION TYPE (path: ~ = reverse direction; rm = topic "
          "removed from answers; F1 = path match on sample)")
    print("%-1s %-36s %5s %-24s %2s %4s %6s  %s"
          % ("h", "type", "n", "path", "rm", "F1", "symb",
             "  ".join("%6s" % c for c in cols[1:])))
    for hop in (1, 2, 3):
        for t in sorted(data[hop][0]):
            v = res[(hop, t)]
            path, rm, fs = paths[(hop, t)]
            worst = min([v["sym"]] + [v[c] for c in cols[1:]])
            if len(data[hop][0]) > 40 and worst >= 0.97 * v["n"]:
                continue
            ps = " > ".join(rname(r) for r in path) if path else "-"
            print("%d %-36s %5d %-24s %2s %4.2f %6.3f  %s"
                  % (hop, t[:36], v["n"], ps, "y" if rm else "n", fs,
                     v["sym"] / v["n"], "  ".join(
                         "%6.3f" % (v[c] / v["n"]) for c in cols[1:])))
    print()
    print("FAN-OUT (objects per subject+relation key) and bank load")
    for L in Ls:
        big = sum(c for c in kc.values() if c > L) / len(S)
        print("  L=%d: facts whose key has more than L objects: %.3f; "
              "fullest bank %d; facts in banks over 2L: %.3f"
              % (L, big, loads[L][0], loads[L][1]))
    for (e, r), c in kc.most_common(5):
        print("  largest key: %-28s %-7s %d objects"
              % (names[e][:28], rname(r), c))
    log("done")


if __name__ == "__main__":
    main()
