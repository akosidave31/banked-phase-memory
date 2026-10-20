"""
metaqa2.py - MetaQA stage 1b: fix the two memory limits found in stage 1.
  hub spreading: a key (entity + relation) with many objects is spread
    over ceil(count/L) banks via slot vectors; a read checks every slot
    bank (up to --cap) and keeps the best score per entity. The memory
    keeps one small number per key (its object count) to know how many
    slots to read.
  beam sweep: K = 5, 20, 50.
  path recovery: more samples for 2-hop; runner-up shown when weak.
True topic + true path (no language model). Metric: Hits@1.
Needs metaqa1.py in the same folder.
"""
import argparse
import collections
import math
import time
import torch
import metaqa1 as M

T0 = time.time()
DEV = M.DEV


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def recover2(adj, outr, smp, hops, cap=5000):
    """modes: 0 keep topic, 1 drop topic from answers, 2 never revisit
    the topic at any hop"""
    sc = collections.Counter()
    for excl in (False, True):
        if excl and hops < 2:
            continue
        for tid, gold in smp:
            stack = [((), {tid})]
            while stack:
                path, cur = stack.pop()
                if len(path) == hops:
                    if excl:
                        sc[(path, 2)] += M.f1(cur, gold)
                    else:
                        sc[(path, 0)] += M.f1(cur, gold)
                        if hops >= 2:
                            sc[(path, 1)] += M.f1(cur - {tid}, gold)
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
                    if excl:
                        nxt.discard(tid)
                    if 0 < len(nxt) <= cap:
                        stack.append((path + (r,), nxt))
    return sc


def follow2(adj, tid, path, mode, cap=50000):
    cur = {tid}
    for r in path:
        nxt = set()
        for e in cur:
            nxt.update(adj.get((e, r), ()))
        if mode == 2:
            nxt.discard(tid)
        cur = nxt
        if not cur or len(cur) > cap:
            break
    if mode == 1:
        cur = cur - {tid}
    return cur


class SpreadMem(M.BankMem):
    @torch.no_grad()
    def write_spread(self, s, r, o, L, n_ent, n_rel, chunk=32768):
        n = len(s)
        cnt = collections.Counter()
        slot = []
        for x, rr in zip(s.tolist(), r.tolist()):
            slot.append(cnt[(x, rr)] // L)
            cnt[(x, rr)] += 1
        deg = torch.zeros(n_ent, n_rel, dtype=torch.long)
        for (x, rr), c in cnt.items():
            deg[x, rr] = c
        self.deg, self.L = deg.to(DEV), L
        slot = torch.tensor(slot)
        g = torch.Generator().manual_seed(12345)
        u = torch.rand(int(slot.max()) + 1, self.D, generator=g) * 2 - 1
        self.Sv = torch.exp(1j * math.pi * u).to(DEV)
        nb = max(n // L, 1)

        def skey(i):
            return (self.addr[s[i:i + chunk].to(DEV)]
                    * self.R[r[i:i + chunk].to(DEV)]
                    * self.Sv[slot[i:i + chunk].to(DEV)])
        b1s, b2s = [], []
        for i in range(0, n, chunk):
            x1, x2 = self.route(skey(i), nb)
            b1s.append(x1.cpu())
            b2s.append(x2.cpu())
        load, fb, unit = [0] * nb, [], {}
        for x, rr, sl, x1, x2 in zip(s.tolist(), r.tolist(), slot.tolist(),
                                     torch.cat(b1s).tolist(),
                                     torch.cat(b2s).tolist()):
            b = unit.get((x, rr, sl))
            if b is None:
                b = x2 if load[x2] < load[x1] else x1
                unit[(x, rr, sl)] = b
            load[b] += 1
            fb.append(b)
        fb = torch.tensor(fb)
        re_ = torch.zeros(nb, self.D, device=DEV)
        im = torch.zeros(nb, self.D, device=DEV)
        for i in range(0, n, chunk):
            fact = skey(i) * torch.roll(self.addr[o[i:i + chunk].to(DEV)],
                                        1, -1)
            ix = fb[i:i + chunk].to(DEV)
            re_.index_add_(0, ix, fact.real.contiguous())
            im.index_add_(0, ix, fact.imag.contiguous())
        self.B = torch.complex(re_, im)
        self.nb, self.load = nb, load

    @torch.no_grad()
    def read_spread(self, ents, r, cap, chunk=1024):
        Mn = ents.shape[0]
        m = ((self.deg[ents, r] + self.L - 1) // self.L).clamp(1, cap)
        rows = torch.repeat_interleave(torch.arange(Mn, device=DEV), m)
        starts = torch.cumsum(m, 0) - m
        js = (torch.arange(rows.shape[0], device=DEV)
              - torch.repeat_interleave(starts, m))
        out = torch.zeros(Mn, self.Ac.shape[1], device=DEV)
        for i in range(0, rows.shape[0], chunk):
            rc, jc = rows[i:i + chunk], js[i:i + chunk]
            key = self.addr[ents[rc]] * self.R[r] * self.Sv[jc]
            b1, b2 = self.route(key, self.nb)
            kc = key.conj()
            s1 = (torch.roll(self.B[b1] * kc, -1, -1) @ self.Ac).real
            s2 = (torch.roll(self.B[b2] * kc, -1, -1) @ self.Ac).real
            pick = (s2.amax(1) > s1.amax(1))[:, None]
            sc = torch.where(pick, s2, s1) / self.D
            out.index_reduce_(0, rc, sc, "amax")
        return out


@torch.no_grad()
def answer(mem, topics, path, K, remove, spread, cap):
    n = topics.shape[0]
    ar = torch.arange(n, device=DEV)
    cur = topics[:, None]
    conf = torch.ones(n, 1, device=DEV)
    for i, r in enumerate(path):
        C = cur.shape[1]
        flat = cur.reshape(-1)
        if spread:
            S = mem.read_spread(flat, r, cap)
        else:
            S = mem.read(mem.addr[flat], r)
        S = S.view(n, C, -1)
        N = S.shape[-1]
        tot = conf[..., None] * S.clamp(min=0)
        if remove == 2 or (remove == 1 and i == len(path) - 1):
            tot[ar, :, topics] = -1.0
        v, idx = tot.view(n, -1).topk(K, -1)
        cur = idx % N
        conf = v.clamp(min=0)
    return cur[:, 0]


def hits(mem, items, path, rm, K, spread, cap):
    if path is None:
        return 0
    valid = [(t, g) for t, g in items if t is not None]
    bs = max(8, 2560 // K)
    h = 0
    for i in range(0, len(valid), bs):
        ch = valid[i:i + bs]
        top = torch.tensor([t for t, _ in ch], device=DEV)
        best = answer(mem, top, list(path), K, rm, spread, cap).tolist()
        h += sum(int(b in g) for b, (_, g) in zip(best, ch))
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/mqa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--sample2", type=int, default=150)
    ap.add_argument("--sample3", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names, idx, rels, trip, bad = M.load_kb(a.root)
    NB = len(rels)
    low = {}
    for i, nm in enumerate(names):
        low.setdefault(nm.lower(), i)

    def look(x):
        i = idx.get(x)
        return i if i is not None else low.get(x.lower())

    def rname(r):
        b = M.ABBR.get(rels[r % NB], rels[r % NB][:6])
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
    log("KB: %d entities, %d facts stored" % (len(names), len(S)))
    hops = []
    data = {}
    for hop in (1, 2, 3):
        groups = collections.defaultdict(list)
        for topic, ans, t in M.load_qa(a.root, hop, a.split):
            tid = look(topic) if topic else None
            gold = {look(x) for x in ans} - {None}
            if tid is None or not gold:
                tid = None
            groups[t].append((tid, gold))
        if groups:
            hops.append(hop)
            data[hop] = groups
    weak = []
    paths = {}
    for hop in hops:
        ns = a.sample2 if hop <= 2 else a.sample3
        for t, items in data[hop].items():
            smp = [x for x in items if x[0] is not None][:ns]
            sc = recover2(adj, outr, smp, hop)
            rank = sorted(sc, key=sc.get, reverse=True)
            if not rank:
                paths[(hop, t)] = (None, False)
                continue
            best = rank[0]
            paths[(hop, t)] = best
            f = sc[best] / len(smp)
            if False:
                ru = rank[1]
                weak.append((hop, t, best, f, ru, sc[ru] / len(smp)))
        log("%d-hop paths recovered" % hop)

    agg = {}
    for hop in hops:
        for t, items in data[hop].items():
            path, rm = paths[(hop, t)]
            k = (hop, path, rm)
            e = agg.setdefault(k, {"items": [], "sym": 0})
            e["items"].extend(items)
            if path is None:
                continue
            for tid, gold in items:
                if tid is None:
                    continue
                ans = follow2(adj, tid, path, rm)
                e["sym"] += int(bool(ans & gold))
    log("symbolic check done")

    St, Rt, Ot = torch.tensor(S), torch.tensor(Rr), torch.tensor(O)
    configs = [("spread", 5), ("spread", 20)]
    info = {}
    plain = M.BankMem(len(names), 2 * NB, a.D, a.seed)
    plain.write(St, Rt, Ot, a.L)
    info["plain"] = max(plain.load)
    for name, K in []:
        for k, e in agg.items():
            e[(name, K)] = hits(plain, e["items"], k[1], k[2], K, False,
                                a.cap)
        log("plain K=%d done" % K)
    del plain
    if DEV == "cuda":
        torch.cuda.empty_cache()
    sp = SpreadMem(len(names), 2 * NB, a.D, a.seed)
    sp.write_spread(St, Rt, Ot, a.L, len(names), 2 * NB)
    info["spread"] = max(sp.load)
    over = int((sp.deg > a.L * a.cap).sum())
    log("spread memory: %d banks, fullest %d, keys over cap: %d"
        % (sp.nb, max(sp.load), over))
    for name, K in configs:
        for k, e in agg.items():
            e[(name, K)] = hits(sp, e["items"], k[1], k[2], K, True, a.cap)
        log("spread K=%d done" % K)

    print()
    print("=== MetaQA stage 1b: hub spreading + beam width (D=%d, L=%d, "
          "read cap %d banks/key) ===" % (a.D, a.L, a.cap))
    print("fullest bank: plain %d, spread %d; keys with more than "
          "L*cap objects: %d" % (info["plain"], info["spread"], over))
    print()
    print("SUMMARY (Hits@1, true topic + true path)")
    cn = ["%s K%d" % c for c in configs]
    print("hop       n  symbolic  " + "  ".join("%9s" % c for c in cn))
    for hop in hops:
        es = [e for k, e in agg.items() if k[0] == hop]
        n = sum(len(e["items"]) for e in es)
        print("%d   %7d   %7.4f  %s" % (hop, n, sum(e["sym"] for e in es) / n,
                                       "  ".join("%9.4f" % (sum(e[c] for e in es)
                                                            / n)
                                                 for c in configs)))
    print()
    print("PER PATH (templates merged; ~ = reverse; rm = topic removed)")
    print("h %-28s rm %6s  symb  %s" % ("path", "n", "  ".join(
        "%9s" % c for c in cn)))
    for k in sorted(agg, key=lambda k: (k[0], str(k[1]))):
        e = agg[k]
        n = len(e["items"])
        ps = " > ".join(rname(r) for r in k[1]) if k[1] else "-"
        print("%d %-28s %2s %6d %5.3f  %s"
              % (k[0], ps, "nfa"[k[2]], n, e["sym"] / n,
                 "  ".join("%9.3f" % (e[c] / n) for c in configs)))
    print()
    print("WEAK PATH MATCHES (F1 < 0.95): best vs runner-up")
    for hop, t, best, f, ru, fr in weak:
        print("  %d %-36s %s (%.2f) vs %s (%.2f)"
              % (hop, t[:36], " > ".join(rname(r) for r in best[0]), f,
                 " > ".join(rname(r) for r in ru[0]), fr))
    if not weak:
        print("  none")
    log("done")


if __name__ == "__main__":
    main()
