"""
diskmem.py - banked phase memory kept on storage, not in RAM. NumPy only,
so it runs in Termux.
  build : read MetaQA kb.txt and write the memory to files in --dir:
          entity phases (1 byte per number, from a hash of the name),
          banks (half precision), key-slot -> bank index, key counts,
          entity type lists; plus a sample of test questions with their
          true relation paths.
  query : in a fresh process, answer the sampled questions by reading only
          the banks each step needs (memory-mapped files); report Hits@1,
          time per question and RAM.
Same design as metaqa2/metaqa3 (one bank per key slot, hub spreading,
beam, never revisit the topic). Clean-up compares only with entities of
the step's target type (films, people, years, ...), read in chunks.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import time
import numpy as np

T0 = time.time()
SM = 1 << 13
TYPE = {"directed_by": "person", "written_by": "person",
        "starred_actors": "person", "release_year": "year",
        "in_language": "language", "has_tags": "tag", "has_genre": "genre",
        "has_imdb_votes": "votes", "has_imdb_rating": "rating"}
_q = np.arange(256)
_q = np.where(_q < 128, _q, _q - 256) * np.pi / 128
COS, SIN = np.cos(_q).astype(np.float32), np.sin(_q).astype(np.float32)
POW2 = (1 << np.arange(31)).astype(np.int64)


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def mem_status():
    out = {}
    try:
        for line in open("/proc/self/status"):
            k, _, v = line.partition(":")
            if k in ("VmRSS", "RssAnon", "RssFile", "VmHWM"):
                out[k] = int(v.split()[0]) / 1024.0
    except OSError:
        pass
    return out


def phasor(p):
    u = np.ascontiguousarray(p).view(np.uint8)
    z = np.empty(u.shape, np.complex64)
    z.real = COS[u]
    z.imag = SIN[u]
    return z


def fixed(seed, n, D):
    return phasor(np.random.default_rng(seed).integers(
        -128, 128, (n, D), dtype=np.int8))


def name_phase(name, D):
    seed = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(seed).integers(-128, 128, D, dtype=np.int8)


def route(key, P1, P2, nb):
    out = []
    for P in (P1, P2):
        bits = ((key @ P.conj().T).real > 0).astype(np.int64)
        out.append((bits @ POW2) % nb)
    return out


def load_kb(root):
    names, idx, rels, rid, trip = [], {}, [], {}, set()

    def eid(x):
        if x not in idx:
            idx[x] = len(names)
            names.append(x)
        return idx[x]
    for line in open(os.path.join(root, "kb.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) != 3:
            continue
        s, r, o = p
        if r not in rid:
            rid[r] = len(rels)
            rels.append(r)
        trip.add((eid(s), rid[r], eid(o)))
    return names, idx, rels, sorted(trip)


def load_qa(root, hop):
    d = os.path.join(root, "%d-hop" % hop)
    for p in (os.path.join(d, "vanilla", "qa_test.txt"),
              os.path.join(d, "qa_test.txt")):
        if os.path.exists(p):
            break
    else:
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        q, _, ans = line.rstrip("\n").partition("\t")
        m = re.search(r"\[(.+?)\]", q)
        out.append((m.group(1) if m else None, ans.split("|"),
                    re.sub(r"\[.+?\]", "[E]", q)))
    return out


def f1(ans, gold):
    inter = len(ans & gold)
    if not inter:
        return 0.0
    p, r = inter / len(ans), inter / len(gold)
    return 2 * p * r / (p + r)


def recover2(adj, outr, smp, hops, cap=5000):
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
                        sc[(path, 2)] += f1(cur, gold)
                    else:
                        sc[(path, 0)] += f1(cur, gold)
                        if hops >= 2:
                            sc[(path, 1)] += f1(cur - {tid}, gold)
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


def build(a):
    os.makedirs(a.dir, exist_ok=True)
    names, idx, rels, trip = load_kb(a.root)
    NE, NB, D = len(names), len(rels), a.D
    NREL = 2 * NB
    low = {}
    for i, n in enumerate(names):
        low.setdefault(n.lower(), i)

    def look(x):
        i = idx.get(x)
        return i if i is not None else low.get(x.lower())
    log("KB: %d entities, %d relations, %d facts" % (NE, NB, len(trip)))

    ph = np.lib.format.open_memmap(os.path.join(a.dir, "entity_phase.npy"),
                                   mode="w+", dtype=np.int8, shape=(NE, D))
    for i, n in enumerate(names):
        ph[i] = name_phase(n, D)
    ph.flush()
    log("entity phases written (from name hashes)")

    S, Rr, O = [], [], []
    for s, r, o in trip:
        S += [s, o]
        Rr += [r, r + NB]
        O += [o, s]
    cnt = collections.Counter()
    J = []
    for s, r in zip(S, Rr):
        J.append(cnt[(s, r)] // a.L)
        cnt[(s, r)] += 1
    max_slot = max(J)
    assert max_slot < SM, max_slot
    deg = np.zeros((NE, NREL), np.int32)
    for (s, r), c in cnt.items():
        deg[s, r] = c
    np.save(os.path.join(a.dir, "deg.npy"), deg)
    nslot = max_slot + 1
    R = fixed(11, NREL, D)
    Sv = fixed(12, nslot, D)
    P1, P2 = fixed(13, 31, D), fixed(14, 31, D)
    n = len(S)
    nb = max(n // a.L, 1)
    S, Rr, O, J = (np.array(x, dtype=np.int64) for x in (S, Rr, O, J))
    b1 = np.empty(n, np.int64)
    b2 = np.empty(n, np.int64)
    ch = 8192
    for i in range(0, n, ch):
        key = phasor(ph[S[i:i + ch]]) * R[Rr[i:i + ch]] * Sv[J[i:i + ch]]
        b1[i:i + ch], b2[i:i + ch] = route(key, P1, P2, nb)
    load, unit, fb = [0] * nb, {}, []
    for s, r, j, x1, x2 in zip(S.tolist(), Rr.tolist(), J.tolist(),
                               b1.tolist(), b2.tolist()):
        b = unit.get((s, r, j))
        if b is None:
            b = x2 if load[x2] < load[x1] else x1
            unit[(s, r, j)] = b
        load[b] += 1
        fb.append(b)
    fb = np.array(fb, dtype=np.int64)
    items = sorted(((s * NREL + r) * SM + j, b) for (s, r, j), b in unit.items())
    np.save(os.path.join(a.dir, "unit_code.npy"),
            np.array([c for c, _ in items], dtype=np.int64))
    np.save(os.path.join(a.dir, "unit_bank.npy"),
            np.array([b for _, b in items], dtype=np.int32))
    log("routing done: %d banks, %d key slots, fullest bank %d"
        % (nb, len(items), max(load)))

    banks = np.lib.format.open_memmap(os.path.join(a.dir, "banks.npy"),
                                      mode="w+", dtype=np.float16,
                                      shape=(nb, 2 * D))
    order = np.argsort(fb, kind="stable")
    fbs = fb[order]
    step = 1024
    for k0 in range(0, nb, step):
        lo = np.searchsorted(fbs, k0)
        hi = np.searchsorted(fbs, k0 + step)
        if lo == hi:
            continue
        ids = order[lo:hi]
        v = (phasor(ph[S[ids]]) * R[Rr[ids]] * Sv[J[ids]]
             * np.roll(phasor(ph[O[ids]]), 1, axis=-1))
        ub, st = np.unique(fbs[lo:hi], return_index=True)
        acc = np.add.reduceat(v, st, axis=0)
        banks[ub, :D] = acc.real.astype(np.float16)
        banks[ub, D:] = acc.imag.astype(np.float16)
        if (k0 // step) % 16 == 0:
            log("  banks %d/%d" % (k0, nb))
    banks.flush()
    log("banks written")

    tname = [TYPE.get(rels[r], rels[r]) if r < NB else "film"
             for r in range(NREL)]
    tset = collections.defaultdict(set)
    for s, r, o in trip:
        tset[tname[r]].add(o)
        tset["film"].add(s)
    tlist = sorted(tset)
    rel_type = np.array([tlist.index(t) for t in tname], dtype=np.int64)
    np.savez(os.path.join(a.dir, "types.npz"), rel_type=rel_type,
             **{"t%d" % i: np.array(sorted(tset[t]), dtype=np.int64)
                for i, t in enumerate(tlist)})

    adj, outr = collections.defaultdict(list), collections.defaultdict(set)
    for s, r, o in trip:
        adj[(s, r)].append(o)
        outr[s].add(r)
        adj[(o, r + NB)].append(s)
        outr[o].add(r + NB)
    rng = random.Random(a.seed)
    rows = []
    for hop in (1, 2, 3):
        qa = load_qa(a.root, hop)
        if not qa:
            continue
        groups = collections.defaultdict(list)
        for topic, ans, t in qa:
            tid = look(topic) if topic else None
            gold = {look(x) for x in ans} - {None}
            if tid is not None and gold:
                groups[t].append((tid, gold))
        ns = a.sample2 if hop <= 2 else a.sample3
        paths = {}
        for k, (t, its) in enumerate(groups.items()):
            sc = recover2(adj, outr, its[:ns], hop)
            if sc:
                paths[t] = max(sc, key=sc.get)
            if (k + 1) % 25 == 0:
                log("  %d-hop paths: %d/%d templates" % (hop, k + 1,
                                                          len(groups)))
        pool = [(t, tid, gold) for t, its in groups.items() if t in paths
                for tid, gold in its]
        for t, tid, gold in rng.sample(pool, min(a.n, len(pool))):
            path, mode = paths[t]
            sym = bool(follow2(adj, tid, path, mode) & gold)
            rows.append((hop, tid, path, mode, sorted(gold), sym))
        log("%d-hop: %d templates, %d questions sampled" % (
            hop, len(paths), min(a.n, len(pool))))
    gold = [g for r in rows for g in r[4]]
    goff = np.cumsum([0] + [len(r[4]) for r in rows])
    np.savez(os.path.join(a.dir, "queries.npz"),
             hop=np.array([r[0] for r in rows]),
             topic=np.array([r[1] for r in rows]),
             path=np.array([list(r[2]) + [-1] * (3 - len(r[2])) for r in rows]),
             mode=np.array([r[3] for r in rows]),
             sym=np.array([r[5] for r in rows]),
             gold=np.array(gold, dtype=np.int64), goff=goff)
    json.dump({"D": D, "L": a.L, "NB": NB, "NREL": NREL, "NE": NE, "nb": nb,
               "nslot": nslot, "rels": rels, "types": tlist,
               "facts": len(trip)},
              open(os.path.join(a.dir, "meta.json"), "w"), indent=1)
    log("build done")


class Ctx:
    def __init__(self, a):
        m = json.load(open(os.path.join(a.dir, "meta.json")))
        self.m = m
        self.D, self.L, self.NREL = m["D"], m["L"], m["NREL"]
        self.ph = np.load(os.path.join(a.dir, "entity_phase.npy"),
                          mmap_mode="r")
        self.banks = np.load(os.path.join(a.dir, "banks.npy"), mmap_mode="r")
        self.deg = np.load(os.path.join(a.dir, "deg.npy"), mmap_mode="r")
        self.codes = np.load(os.path.join(a.dir, "unit_code.npy"),
                             mmap_mode="r")
        self.ubank = np.load(os.path.join(a.dir, "unit_bank.npy"),
                             mmap_mode="r")
        T = np.load(os.path.join(a.dir, "types.npz"))
        self.rel_type = T["rel_type"]
        self.types = [T["t%d" % i] for i in range(len(m["types"]))]
        self.R = fixed(11, self.NREL, self.D)
        self.Sv = fixed(12, m["nslot"], self.D)
        self.K, self.cap, self.chunk = a.K, a.cap, a.chunk


def answer(topic, path, mode, c):
    D, L, NREL = c.D, c.L, c.NREL
    cand = np.array([topic], dtype=np.int64)
    conf = np.ones(1, np.float32)
    for i, r in enumerate(path):
        vocab = c.types[c.rel_type[r]]
        V = len(vocab)
        tot = np.zeros((len(cand), V), np.float32)
        ns = np.minimum((np.asarray(c.deg[cand, r], dtype=np.int64) + L - 1)
                        // L, c.cap)
        if ns.sum() > 0:
            ent = np.repeat(cand, ns)
            starts = np.cumsum(ns) - ns
            js = np.arange(ns.sum()) - np.repeat(starts, ns)
            code = (ent * NREL + r) * SM + js
            ix = np.minimum(np.searchsorted(c.codes, code), len(c.codes) - 1)
            ok = np.asarray(c.codes[ix]) == code
            bank = np.asarray(c.ubank[ix])
            key = phasor(c.ph[ent]) * c.R[r] * c.Sv[js]
            Bv = np.asarray(c.banks[bank], dtype=np.float32)
            Bc = np.empty((len(ent), D), np.complex64)
            Bc.real = Bv[:, :D]
            Bc.imag = Bv[:, D:]
            v = np.roll(Bc * np.conj(key), -1, axis=1)
            v[~ok] = 0
            vr = np.ascontiguousarray(v.real)
            vi = np.ascontiguousarray(v.imag)
            has = np.flatnonzero(ns > 0)
            seg = starts[has]
            for c0 in range(0, V, c.chunk):
                u = np.asarray(c.ph[vocab[c0:c0 + c.chunk]]).view(np.uint8)
                sc = (vr @ COS[u].T + vi @ SIN[u].T) / D
                tot[has, c0:c0 + u.shape[0]] = np.maximum.reduceat(
                    sc, seg, axis=0)
        tot = conf[:, None] * np.clip(tot, 0, None)
        if mode == 2 or (mode == 1 and i == len(path) - 1):
            p = np.searchsorted(vocab, topic)
            if p < V and vocab[p] == topic:
                tot[:, p] = -1.0
        flat = tot.ravel()
        k = min(c.K, flat.size)
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.argsort(-flat[top])]
        cand = vocab[top % V]
        conf = flat[top]
    return int(cand[0])


def query(a):
    m0 = mem_status()
    c = Ctx(a)
    Q = np.load(os.path.join(a.dir, "queries.npz"))
    hop, topic, path = Q["hop"], Q["topic"], Q["path"]
    mode, sym, gold, goff = Q["mode"], Q["sym"], Q["gold"], Q["goff"]
    m1 = mem_status()
    log("loaded: RAM %.0f MB private (%.0f MB before loading the memory)"
        % (m1.get("RssAnon", 0), m0.get("RssAnon", 0)))
    res, tms = collections.defaultdict(list), collections.defaultdict(list)
    peak_anon = peak_file = 0.0
    n = len(hop)
    for i in range(n):
        p = [r for r in path[i].tolist() if r >= 0]
        t0 = time.time()
        best = answer(int(topic[i]), p, int(mode[i]), c)
        tms[int(hop[i])].append(time.time() - t0)
        res[int(hop[i])].append(bool(np.isin(best, gold[goff[i]:goff[i + 1]])))
        ms = mem_status()
        peak_anon = max(peak_anon, ms.get("RssAnon", 0))
        peak_file = max(peak_file, ms.get("RssFile", 0))
        if (i + 1) % 25 == 0 or i + 1 == n:
            el = time.time() - T0
            log("%d/%d  %s  %.2f s/question  RAM %.0f MB private (+%.0f MB "
                "file cache)  ETA %.0f min"
                % (i + 1, n, "  ".join("%d-hop %.3f" % (h, np.mean(v))
                                       for h, v in sorted(res.items())),
                   np.mean([t for v in tms.values() for t in v]),
                   ms.get("RssAnon", 0), ms.get("RssFile", 0),
                   el / (i + 1) * (n - i - 1) / 60))
    sizes = {f: os.path.getsize(os.path.join(a.dir, f)) / 2 ** 20 for f in
             ("entity_phase.npy", "banks.npy", "deg.npy", "unit_code.npy",
              "unit_bank.npy", "types.npz")}
    print()
    print("=== Banked phase memory on storage: MetaQA, true topic + true "
          "path (D=%d, beam %d) ===" % (c.D, c.K))
    print("memory: %d facts (%d with inverses) in %d banks"
          % (c.m["facts"], 2 * c.m["facts"], c.m["nb"]))
    print("files on storage: " + ", ".join("%s %.1f MB" % kv
                                           for kv in sizes.items())
          + "; total %.0f MB" % sum(sizes.values()))
    print("RAM, private (RssAnon): before loading %.0f MB, after loading "
          "%.0f MB, peak %.0f MB" % (m0.get("RssAnon", 0),
                                     m1.get("RssAnon", 0), peak_anon))
    print("RAM, file cache (RssFile, reclaimable by the OS): peak %.0f MB"
          % peak_file)
    print()
    print("hop     n  Hits@1  symbolic  s/question (mean, max)")
    for h in sorted(res):
        sel = hop == h
        print("%d   %5d  %6.3f    %6.3f   %.2f, %.2f"
              % (h, len(res[h]), np.mean(res[h]), sym[sel].mean(),
                 np.mean(tms[h]), np.max(tms[h])))
    print("reference, full-RAM GPU version on all test questions (stage "
          "1b): 2-hop 0.984, 3-hop 0.993; its memory used ~1.26 GB of RAM")
    log("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "query"])
    ap.add_argument("--root", default="~/mqa")
    ap.add_argument("--dir", default="~/dmem")
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sample2", type=int, default=150)
    ap.add_argument("--sample3", type=int, default=40)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.root = os.path.expanduser(a.root)
    a.dir = os.path.expanduser(a.dir)
    build(a) if a.cmd == "build" else query(a)


if __name__ == "__main__":
    main()
