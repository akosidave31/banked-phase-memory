"""
freqmem_kg11.py - Experiment 11: robustness on FB15k-237.
Routing id (hashed IDs) vs vec (sign bits of random projections of
the key vector E_h*R_r, no IDs). 5 seeds, D via --D, 20 chains
(max 2 per first relation), beam K = 1, 2, 5, 10. Graph-correct and
filtered accuracy with their chance levels; per-chain; symbolic ref.
Pre-registered bar: vec L=16, beam 5: graph-correct >= 0.90,
spread (std over seeds) <= 0.02.
"""
import argparse
import collections
import json
import math
import os
import random
import statistics
import time
import urllib.request
import torch

T0 = time.time()
URLS = [
    "https://raw.githubusercontent.com/DeepGraphLearning/"
    "KnowledgeGraphEmbedding/master/data/FB15k-237/train.txt",
    "https://huggingface.co/datasets/KGraph/FB15k-237/"
    "resolve/main/data/train.txt",
]


def log(msg):
    t = time.time() - T0
    print("[%6.0fs] %s" % (t, msg), flush=True)


def download(path):
    if os.path.exists(path) and os.path.getsize(path) > 1000000:
        return path
    for u in URLS:
        try:
            log("downloading " + u)
            urllib.request.urlretrieve(u, path)
            if os.path.getsize(path) > 1000000:
                return path
        except Exception as e:
            log("  failed: %s" % e)
    raise SystemExit("Download failed; upload train.txt as " + path)


def load(path):
    ents, rels, trip = {}, {}, []
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) != 3:
            p = line.split()
        if len(p) != 3:
            continue
        h = ents.setdefault(p[0], len(ents))
        r = rels.setdefault(p[1], len(rels))
        t = ents.setdefault(p[2], len(ents))
        trip.append((h, r, t))
    return trip, ents, rels


def build_graph(trip):
    dd = collections.defaultdict
    tails, nbr = dd(list), dd(set)
    in_rels, out_rels, rel_heads = dd(set), dd(set), dd(set)
    for h, r, t in trip:
        tails[(h, r)].append(t)
        nbr[h].add(t)
        nbr[t].add(h)
        out_rels[h].add(r)
        in_rels[t].add(r)
        rel_heads[r].add(h)
    return tails, nbr, in_rels, out_rels, rel_heads


def pick_chains(n_ent, in_rels, out_rels, k, per_r1, min_sup=100):
    sup = collections.Counter()
    for m in range(n_ent):
        for r1 in in_rels.get(m, ()):
            for r2 in out_rels.get(m, ()):
                sup[(r1, r2)] += 1
    out, used = [], collections.Counter()
    for c, s in sup.most_common():
        if s < min_sup:
            break
        if used[c[0]] >= per_r1:
            continue
        out.append((c, s))
        used[c[0]] += 1
        if len(out) >= k:
            break
    return out


def two_hop_queries(chains, tails, nbr, rel_heads, per_chain, rng):
    out = []
    for ci, ((r1, r2), _) in enumerate(chains):
        heads = sorted(rel_heads[r1])
        rng.shuffle(heads)
        n = 0
        for h in heads:
            mids = set(tails[(h, r1)])
            full = set()
            for m in mids:
                full.update(tails.get((m, r2), ()))
            A = full - nbr[h]
            A.discard(h)
            if not A:
                continue
            m_star = None
            for m in mids:
                if set(tails.get((m, r2), ())) & A:
                    m_star = m
                    break
            out.append(dict(h=h, r1=r1, r2=r2, A=A, full=full,
                            mids=mids, m_star=m_star, chain=ci))
            n += 1
            if n >= per_chain:
                break
    return out


def symbolic_vote(q, tails):
    cnt = collections.Counter()
    for m in q["mids"]:
        cnt.update(tails.get((m, q["r2"]), ()))
    return cnt.most_common(1)[0][0]


def frac(flags):
    flags = list(flags)
    return sum(flags) / len(flags)


class Memory:
    BITS = 31

    def __init__(self, trip, n_ent, n_rel, D, dev, seed, routing):
        g = torch.Generator().manual_seed(seed)

        def ph(n):
            u = torch.rand(n, D, generator=g) * 2 - 1
            return torch.exp(1j * u * math.pi).to(dev)
        self.E, self.R = ph(n_ent), ph(n_rel)
        self.P1, self.P2 = ph(self.BITS), ph(self.BITS)
        ar = torch.arange(self.BITS, device=dev)
        self.pow2 = (2 ** ar).long()
        self.Ec = self.E.conj().T.contiguous()
        T = torch.tensor(trip, device=dev)
        self.H, self.Rr, self.Tt = T[:, 0], T[:, 1], T[:, 2]
        self.trip, self.D, self.dev = trip, D, dev
        self.seed, self.routing = seed, routing

    def route(self, h, r, nb):
        if self.routing == "id":
            pairs = list(zip(h.tolist(), r.tolist()))
            s = self.seed
            b1 = [hash((x, y, 1, s)) % nb for x, y in pairs]
            b2 = [hash((x, y, 2, s)) % nb for x, y in pairs]
            dv = self.dev
            return (torch.tensor(b1, device=dv),
                    torch.tensor(b2, device=dv))
        key = self.E[h] * self.R[r]
        out = []
        for P in (self.P1, self.P2):
            bits = ((key @ P.conj().T).real > 0).long()
            out.append((bits * self.pow2).sum(1) % nb)
        return out[0], out[1]

    def facts(self, sl):
        a = self.E[self.H[sl]] * self.R[self.Rr[sl]]
        return a * torch.roll(self.E[self.Tt[sl]], 1, -1)

    def build_single(self, chunk=32768):
        M = torch.zeros(self.D, dtype=torch.complex64,
                        device=self.dev)
        for i in range(0, len(self.H), chunk):
            M += self.facts(slice(i, i + chunk)).sum(0)
        return {"kind": "single", "M": M, "size": self.D}

    def build_banks(self, tails, L, chunk=32768):
        F = len(self.H)
        nb = max(F // L, 1)
        keys = sorted(tails.keys())
        dv = self.dev
        kh = torch.tensor([k[0] for k in keys], device=dv)
        kr = torch.tensor([k[1] for k in keys], device=dv)
        c1, c2 = [], []
        for i in range(0, len(keys), 65536):
            x, y = self.route(kh[i:i + 65536], kr[i:i + 65536], nb)
            c1.append(x)
            c2.append(y)
        c1 = torch.cat(c1).tolist()
        c2 = torch.cat(c2).tolist()
        order = list(range(len(keys)))
        random.Random(self.seed).shuffle(order)
        load = [0] * nb
        where = {}
        for i in order:
            x, y = c1[i], c2[i]
            b = y if load[y] < load[x] else x
            where[keys[i]] = b
            load[b] += len(tails[keys[i]])
        fb = [where[(h, r)] for h, r, _ in self.trip]
        fb = torch.tensor(fb, device=dv)
        re = torch.zeros(nb, self.D, device=dv)
        im = torch.zeros(nb, self.D, device=dv)
        for i in range(0, F, chunk):
            f = self.facts(slice(i, i + chunk))
            idx = fb[i:i + chunk]
            re.index_add_(0, idx, f.real.contiguous())
            im.index_add_(0, idx, f.imag.contiguous())
        return {"kind": "banks", "B": torch.complex(re, im),
                "nb": nb, "size": nb * self.D}

    @torch.no_grad()
    def scores(self, mem, hs, rs):
        h = torch.tensor(hs, device=self.dev)
        r = torch.tensor(rs, device=self.dev)
        key = (self.E[h] * self.R[r]).conj()
        if mem["kind"] == "single":
            est = torch.roll(mem["M"][None] * key, -1, -1)
            return (est @ self.Ec).real
        b1, b2 = self.route(h, r, mem["nb"])
        e1 = torch.roll(mem["B"][b1] * key, -1, -1)
        e2 = torch.roll(mem["B"][b2] * key, -1, -1)
        s1 = (e1 @ self.Ec).real
        s2 = (e2 @ self.Ec).real
        pick2 = (s2.amax(1) > s1.amax(1))[:, None]
        return torch.where(pick2, s2, s1)

    def top(self, mem, hs, rs, k, bs=256):
        vs, js = [], []
        for i in range(0, len(hs), bs):
            s = self.scores(mem, hs[i:i + bs], rs[i:i + bs])
            v, j = s.topk(k, dim=1)
            vs.append(v)
            js.append(j)
        return torch.cat(vs), torch.cat(js)


def evaluate(mo, mem, atomic, probe, two, beams):
    hs = [q[0] for q in atomic]
    rs = [q[1] for q in atomic]
    _, pa = mo.top(mem, hs, rs, 1)
    pa = pa[:, 0].tolist()
    atom = frac(p in q[2] for p, q in zip(pa, atomic))
    hs = [q[0] for q in probe]
    rs = [q[1] for q in probe]
    _, pp = mo.top(mem, hs, rs, 1)
    pp = pp[:, 0].tolist()
    hop2 = frac(p in q[2] for p, q in zip(pp, probe))
    K = max(beams)
    hs = [q["h"] for q in two]
    rs = [q["r1"] for q in two]
    v1, m1 = mo.top(mem, hs, rs, K)
    fm = m1.reshape(-1).tolist()
    fr = [q["r2"] for q in two for _ in range(K)]
    v2, t2 = mo.top(mem, fm, fr, 1)
    v2, t2 = v2.view(-1, K), t2.view(-1, K)
    ar = torch.arange(len(two), device=t2.device)
    res = {"atomic": atom, "hop2fact": hop2}
    kc = 5 if 5 in beams else K
    for k in beams:
        cf = v1[:, :k].clamp(min=0) * v2[:, :k].clamp(min=0)
        pb = t2[ar, cf.argmax(1)].tolist()
        ok = [t in q["full"] for t, q in zip(pb, two)]
        res["full%d" % k] = frac(ok)
        res["filt%d" % k] = frac(t in q["A"] for t, q in zip(pb, two))
        if k == kc:
            per = collections.defaultdict(list)
            for o, q in zip(ok, two):
                per[q["chain"]].append(o)
            res["per_chain"] = {ci: frac(v) for ci, v in per.items()}
    return res


def report(results, beams, labels, n_per, ch_full, ch_filt, vote, a):
    names = list(dict.fromkeys(r["memory"] for r in results))

    def agg(name, key):
        v = [r[key] for r in results if r["memory"] == name]
        s = statistics.pstdev(v) if len(v) > 1 else 0.0
        return statistics.mean(v), s

    def ms(name, key):
        m, s = agg(name, key)
        return ("%.3f+-%.3f" % (m, s)).rjust(12)

    kc = 5 if 5 in beams else max(beams)
    k0 = beams[0]
    print()
    print("=== EXP 11: FB15k-237 robustness, D=%d, %d seeds, "
          "%d chains (max %d per first relation) ==="
          % (a.D, a.seeds, len(labels), a.per_r1))
    print("chance: graph-correct ~%.4f, filtered ~%.4f"
          % (ch_full, ch_filt))
    print("symbolic lookup: graph-correct 1.000; its most common "
          "answer survives the filter %.3f of the time" % vote)
    print()
    cols = ["atomic", "hop2fact", "full%d" % k0, "full%d" % kc,
            "filt%d" % kc]
    heads = ["atomic", "hop2fact", "greedy", "beam%d" % kc,
             "filt%d" % kc]
    print("memory".ljust(10) + "numbers".rjust(12)
          + "".join(h.rjust(12) for h in heads))
    for n in names:
        size = [r["size"] for r in results if r["memory"] == n][0]
        print(n.ljust(10) + ("{:,}".format(size)).rjust(12)
              + "".join(ms(n, c) for c in cols))
    print()
    print("BEAM SWEEP (graph-correct)")
    print("memory".ljust(10)
          + "".join(("K=%d" % k).rjust(12) for k in beams))
    for n in names:
        print(n.ljust(10)
              + "".join(ms(n, "full%d" % k) for k in beams))
    target = "vec L=16" if "vec L=16" in names else names[-1]
    print()
    print("PER CHAIN (%s, beam %d, graph-correct)" % (target, kc))
    for ci, lab in enumerate(labels):
        v = [r["per_chain"].get(ci, float("nan"))
             for r in results if r["memory"] == target]
        print("  %-45s n=%4d acc=%.3f"
              % (lab, n_per[ci], statistics.mean(v)))
    if "vec L=16" in names:
        m, s = agg("vec L=16", "full%d" % kc)
        ok = m >= 0.90 and s <= 0.02
        print()
        print("PRE-REGISTERED BAR (vec L=16, beam %d): %.3f+-%.3f"
              " -> %s (needs >= 0.90 and spread <= 0.02)"
              % (kc, m, s, "PASS" if ok else "FAIL"))
    for r in results:
        pc = r["per_chain"]
        r["per_chain"] = {str(k): v for k, v in pc.items()}
    fn = "kg11_results_D%d.json" % a.D
    with open(fn, "w") as f:
        json.dump({"chains": labels, "results": results}, f,
                  indent=1)
    log("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--banks", default="64,16,4")
    ap.add_argument("--routing", default="id,vec")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--chains", type=int, default=20)
    ap.add_argument("--per_r1", type=int, default=2)
    ap.add_argument("--per_chain", type=int, default=300)
    ap.add_argument("--beams", default="1,2,5,10")
    ap.add_argument("--n_atomic", type=int, default=3000)
    ap.add_argument("--data", default="fb15k237_train.txt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    beams = [int(x) for x in a.beams.split(",")]
    routings = a.routing.split(",")
    trip, ents, rels = load(download(a.data))
    n_ent, n_rel = len(ents), len(rels)
    rname = {v: k for k, v in rels.items()}

    def short(r):
        return rname[r].split("/")[-1]
    log("device=%s D=%d: %d facts, %d entities, %d relations"
        % (dev, a.D, len(trip), n_ent, n_rel))
    tails, nbr, in_rels, out_rels, rel_heads = build_graph(trip)
    chains = pick_chains(n_ent, in_rels, out_rels, a.chains,
                         a.per_r1)
    labels = [short(r1) + "->" + short(r2)
              for (r1, r2), _ in chains]
    for lab, (_, s) in zip(labels, chains):
        log("  chain %s (support %d)" % (lab, s))
    rng = random.Random(0)
    two = two_hop_queries(chains, tails, nbr, rel_heads,
                          a.per_chain, rng)
    atomic = [(h, r, set(tails[(h, r)]))
              for h, r, _ in rng.sample(trip, a.n_atomic)]
    probe = [(q["m_star"], q["r2"],
              set(tails[(q["m_star"], q["r2"])])) for q in two]
    nq = len(two)
    ch_full = sum(len(q["full"]) for q in two) / nq / n_ent
    ch_filt = sum(len(q["A"]) for q in two) / nq / n_ent
    vote = frac(symbolic_vote(q, tails) in q["A"] for q in two)
    n_per = collections.Counter(q["chain"] for q in two)
    log("%d two-hop queries over %d chains" % (nq, len(chains)))
    results = []
    k0, kmax = beams[0], max(beams)
    for seed in range(a.seeds):
        for ri, routing in enumerate(routings):
            mo = Memory(trip, n_ent, n_rel, a.D, dev, seed, routing)
            fmts = [("single", None)] if ri == 0 else []
            for L in a.banks.split(","):
                fmts.append(("%s L=%s" % (routing, L), int(L)))
            for name, L in fmts:
                if L is None:
                    mem = mo.build_single()
                else:
                    mem = mo.build_banks(tails, L)
                r = evaluate(mo, mem, atomic, probe, two, beams)
                r.update(seed=seed, memory=name, size=mem["size"])
                results.append(r)
                log("  seed %d %-10s atomic=%.3f hop2=%.3f "
                    "greedy=%.3f beam%d=%.3f"
                    % (seed, name, r["atomic"], r["hop2fact"],
                       r["full%d" % k0], kmax, r["full%d" % kmax]))
                del mem
                torch.cuda.empty_cache()
            del mo
    report(results, beams, labels, n_per, ch_full, ch_filt, vote, a)


if __name__ == "__main__":
    main()
