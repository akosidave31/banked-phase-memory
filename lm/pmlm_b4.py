"""
pmlm_b4.py - Track B: why do unseen relation families fail, and does
more training variety fix it?
Both arms are tested on the SAME held-out families of relations_v1.json:
  v1  : slots parser trained on the v1 training families (as b2/b3)
  new : also trained on relations_extra.json (new families only; any
        phrase sharing a word stem with a held-out family is dropped)
Per seed and arm: answers on conditions A, B, D (same questions as
pmlm_b3), plus a breakdown of condition B:
  - why each question failed (name, hop count, relation, memory)
  - hop-count confusion (true vs predicted)
  - relation accuracy for every held-out phrase
Needs pmlm_b.py, pmlm_b2.py, relations_v1.json, relations_extra.json.
"""
import argparse
import collections
import json
import random
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pmlm_b as B
import pmlm_b2 as B2

T0 = time.time()
DEV = B.DEV
NR, HMAX = B.NR, B.HMAX
REL = B2.REL_NAMES
CONDS = [("A", "seen fam   + seen openers", 0),
         ("B", "UNSEEN fam + seen openers", 1),
         ("D", "UNSEEN fam + UNSEEN openers", 3)]
CAUSES = ["correct", "name", "hop-count", "relation", "memory"]


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def words(t):
    return re.findall(r"[a-z]+", t.replace("{X}", " ").lower())


def add_extra(tr, held, v1_path, ex_path):
    v1 = json.load(open(v1_path))
    known = {it["text"].strip() for d in v1.values()
             for items in d.values() for it in items}
    stems = set()
    for fams in held.values():
        for f in fams:
            stems.add(f.lower())
            if f.endswith("e"):
                stems.add(f[:-1].lower())
    pool = {k: [list(x) for x in tr[k]] for k in ("np", "q")}
    added = {k: [0] * NR for k in ("np", "q")}
    dropped = []
    ex = json.load(open(ex_path))
    for r, name in enumerate(REL):
        for kind, key in (("noun_phrases", "np"), ("questions", "q")):
            for it in ex.get(name, {}).get(kind, []):
                t = it["text"].strip()
                if key == "np" and "{X}" in t:
                    continue
                if key == "q" and t.count("{X}") != 1:
                    continue
                hit = [w for w in words(t)
                       if any(w.startswith(s) for s in stems)]
                if hit:
                    dropped.append("%s: '%s' (%s)" % (name, t, hit[0]))
                    continue
                if t in known or t in pool[key][r]:
                    continue
                pool[key][r].append(t)
                added[key][r] += 1
    return pool, added, dropped


def held_list(v1_path, te):
    v1 = json.load(open(v1_path))
    out = []
    for r, name in enumerate(REL):
        fam = {it["text"].strip(): it["family"]
               for it in v1[name]["noun_phrases"]}
        for t in te["np"][r]:
            out.append((r, fam.get(t, "?"), t))
    return out


@torch.no_grad()
def run_eval(m, lm, tok, mem, addr, W, qs, a, bs=64):
    recs = []
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        names, rels, hops = B2.parse_slots(m, lm, tok, ch, a.k_layers)
        sa = torch.stack([addr.name(nm) for nm in names]).to(DEV)
        best, unk = B.beam_answer(mem, W, sa, rels, hops, a.K, a.tau)
        for q, nm, rr, ph, b, u in zip(ch, names, rels.tolist(),
                                       hops.tolist(), best.tolist(),
                                       unk.tolist()):
            h = len(q["rels"])
            if not q["gold"]:
                ok = bool(u)
            else:
                ok = (not u) and b in q["gold"]
            if ok:
                cause = "correct"
            elif nm != W.name[q["s"]]:
                cause = "name"
            elif ph != h:
                cause = "hop-count"
            elif rr[:h] != q["rels"]:
                cause = "relation"
            else:
                cause = "memory"
            recs.append(dict(q=q, h=h, ph=ph, rr=rr, cause=cause))
    return recs


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="relations_v1.json")
    ap.add_argument("--extra", default="relations_extra.json")
    ap.add_argument("--arms", default="v1,new")
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
    pools, added, dropped = {"v1": tr}, None, []
    if "new" in arms:
        pools["new"], added, dropped = add_extra(tr, held, a.data,
                                                 a.extra)
        log("new arm: +%d noun phrases, +%d questions, %d dropped"
            % (sum(added["np"]), sum(added["q"]), len(dropped)))
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
        lambda: [0, 0, collections.Counter()]) for arm in arms}
    for seed in range(a.seeds):
        for arm in arms:
            torch.manual_seed(seed)
            m = B.Parser(hid, maxlen=96).to(DEV)
            log("seed %d arm %s: training slots parser" % (seed, arm))
            B2.train("slots", m, lm, tok, a, random.Random(seed + 1),
                     pools[arm])
            for c, _, _ in CONDS:
                per_h = []
                for h in range(HMAX):
                    recs = run_eval(m, lm, tok, mem, addr, W,
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
    print("=== Track B v4: why unseen families fail (%s, %d seeds per "
          "arm) ===" % (a.lm, a.seeds))
    print("held-out families: " + "; ".join(
        "%s: %s" % (k, ", ".join(v)) for k, v in held.items()))
    if added:
        print("new arm adds (noun phrases/questions): " + ", ".join(
            "%s +%d/+%d" % (REL[r], added["np"][r], added["q"][r])
            for r in range(NR)))
        print("dropped by stem guard: %d" % len(dropped))
        for d in dropped:
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
    print()
    print("RELATION ACCURACY PER HELD-OUT PHRASE, condition B "
          "(seeds pooled)")
    print("%-10s %-8s %-35s %5s  %s  most common wrong guess"
          % ("relation", "family", "phrase", "n",
             "  ".join("%5s" % arm for arm in arms)))
    for r, fam, t in held_list(a.data, te):
        st = [phr[arm].get((r, t)) for arm in arms]
        n = st[0][0] if st[0] else 0
        accs = "  ".join("%5.3f" % (s[1] / s[0]) if s and s[0]
                         else "    -" for s in st)
        wrong = " / ".join(REL[s[2].most_common(1)[0][0]]
                           if s and s[2] else "-" for s in st)
        print("%-10s %-8s %-35s %5d  %s  %s"
              % (REL[r], fam, t, n, accs, wrong))
    log("done")


if __name__ == "__main__":
    main()
