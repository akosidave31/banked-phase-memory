"""
pmlm_b6.py - Track B v6: let the frozen LM name the relations.
The zero-shot test showed Qwen already knows every held-out phrase
(typed: 1.000) while the trained parser misread several. Variants:
  slots          : slots parser (v1 data), as before
  slots+schema   : same parser, relations constrained by the schema
  tagger+qwen    : a small tagger finds the name and splits the
                   question into phrases (hop count = number of
                   phrases); Qwen picks each phrase's relation
                   zero-shot among the relations the schema allows
  gold-seg+qwen  : true name and phrases, Qwen relations (ceiling,
                   no training)
Schema: a relation must fit the current entity's type (person or
company); a hop that is not the last must lead to a person or
company (a city has no outgoing relations).
All training uses v1 data only. Needs pmlm_b.py, pmlm_b2.py,
pmlm_b4.py, pmlm_zs.py, relations_v1.json.
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
import pmlm_zs as ZS

T0 = time.time()
DEV = B.DEV
NR, HMAX = B.NR, B.HMAX
REL = B2.REL_NAMES
CONDS = [("A", 0), ("B", 1), ("D", 3)]
CAUSES = ["correct", "name", "hop-count", "relation", "memory"]
VARS = ["slots", "slots+schema", "tagger+qwen", "gold-seg+qwen"]


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def allowed(t, last):
    if t not in (0, 1):
        return list(range(NR))
    return list(B.OUT[t] if last else B.CONT[t])


def gold_phrases(q):
    return [q["text"][x:y] for x, y, _ in sorted(q["segs"],
                                                  key=lambda s: s[2])]


class BIOTagger(nn.Module):
    """Tags: 0 other, 1 name, 2 phrase start, 3 phrase inside."""

    def __init__(self, hid, d=256, layers=2, maxlen=96):
        super().__init__()
        self.proj = nn.Linear(hid, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(
            d, 4, 4 * d, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers,
                                         enable_nested_tensor=False)
        self.out = nn.Linear(d, 4)

    def forward(self, f, mask):
        T = f.shape[1]
        h = self.proj(f) + self.pos.weight[None, :T]
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.out(h)


def bio_targets(enc, qs):
    tags = B2.tag_targets(enc, qs)
    out = torch.zeros_like(tags)
    for b in range(tags.shape[0]):
        seen = set()
        for j, v in enumerate(tags[b].tolist()):
            if v == 1:
                out[b, j] = 1
            elif v >= 2:
                out[b, j] = 3 if v in seen else 2
                seen.add(v)
    return out


def train_tagger(m, lm, tok, a, rng, pool):
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    every = max(a.steps // 6, 1)
    world = None
    for step in range(1, a.steps + 1):
        if step == 1 or step % 50 == 0:
            world = B.World(a.n_train, rng)
        qs = [B2.train_q(world, rng, pool) for _ in range(a.bs)]
        enc = B.tokenize(tok, qs)
        mask = enc["attention_mask"].bool().to(DEV)
        f = B2.feats(lm, enc, a.k_layers)
        tg = bio_targets(enc, qs).to(DEV).masked_fill(~mask, -100)
        lg = m(f, mask)
        loss = F.cross_entropy(lg.reshape(-1, 4), tg.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % every == 0:
            acc = (lg.argmax(-1) == tg)[mask].float().mean().item()
            log("  tagger step %d loss=%.3f acc=%.3f"
                % (step, loss.item(), acc))


@torch.no_grad()
def parse_bio(m, lm, tok, qs, k, bs=64):
    m.eval()
    out = []
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        enc = B.tokenize(tok, ch)
        mask = enc["attention_mask"].bool().to(DEV)
        pt = m(B2.feats(lm, enc, k), mask).argmax(-1)
        pt = pt.masked_fill(~mask, 0).tolist()
        offs = enc["offset_mapping"].tolist()
        for b, q in enumerate(ch):
            ni = [j for j, v in enumerate(pt[b]) if v == 1]
            if ni:
                nm = B2.clean(q["text"][offs[b][ni[0]][0]:
                                        offs[b][ni[-1]][1]])
                stop = ni[0]
            else:
                nm, stop = "", len(pt[b])
            segs = []
            for j in range(stop):
                v = pt[b][j]
                if v == 2 or (v == 3 and not segs):
                    segs.append([offs[b][j][0], offs[b][j][1]])
                elif v == 3:
                    segs[-1][1] = offs[b][j][1]
            ph = [q["text"][x:y].strip() for x, y in segs]
            out.append((nm, ph[::-1]))
    m.train()
    return out


def schema_decode(row, hops, t0):
    rels, t = [], t0
    for i in range(HMAX):
        opts = allowed(t, i == hops - 1) if i < hops else range(NR)
        r = max(opts, key=lambda x: row[i][x])
        if i < hops:
            t = B.RNG[r]
        rels.append(r)
    return rels


@torch.no_grad()
def slots_parses(m, lm, tok, qs, k, W, widx, bs=64):
    plain, schema = [], []
    m.eval()
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        enc = B.tokenize(tok, ch)
        mask = enc["attention_mask"].bool().to(DEV)
        st, en, rl, hp = m(B2.feats(lm, enc, k), mask)
        T = st.shape[1]
        ii = torch.arange(T, device=DEV)
        valid = ((ii[None, :] >= ii[:, None])
                 & (ii[None, :] - ii[:, None] <= 12))
        sc = (st[:, :, None] + en[:, None, :]).masked_fill(~valid[None],
                                                           -1e9)
        flat = sc.reshape(len(ch), -1).argmax(-1)
        s_i, e_i = (flat // T).tolist(), (flat % T).tolist()
        offs = enc["offset_mapping"]
        rl = rl.float().cpu().tolist()
        hops = (hp.argmax(-1) + 1).tolist()
        for b, q in enumerate(ch):
            x0, x1 = int(offs[b, s_i[b], 0]), int(offs[b, e_i[b], 1])
            nm = B2.clean(q["text"][x0:x1])
            row = rl[b]
            plain.append((nm, [max(range(NR), key=lambda x: row[j][x])
                               for j in range(HMAX)], hops[b]))
            t0 = W.type[widx[nm]] if nm in widx else None
            schema.append((nm, schema_decode(row, hops[b], t0), hops[b]))
    m.train()
    return plain, schema


class QwenRel:
    """Zero-shot relation choice by the frozen LM, cached by
    (phrase, allowed relations)."""

    def __init__(self, lm, tok, dummy, bs=32):
        self.lm, self.tok, self.dummy, self.bs = lm, tok, dummy, bs
        self.cache = {}

    def resolve(self, keys):
        todo = sorted({k for k in keys if k not in self.cache})
        jobs = []
        for k in todo:
            ph, opts = k
            if len(opts) == 1:
                self.cache[k] = opts[0]
                continue
            doms = {B.DOM[r] for r in opts}
            names = self.dummy[doms.pop() if len(doms) == 1 else 0]
            for nm in names:
                for i in range(len(opts)):
                    rot = list(opts[i:] + opts[:i])
                    jobs.append((k, rot, ZS.prompt(self.tok, ph, nm, rot)))
        if not jobs:
            return
        lps = ZS.score(self.lm, self.tok, [j[2] for j in jobs],
                       [len(j[1]) for j in jobs], self.bs)
        sc = collections.defaultdict(lambda: [0.0] * NR)
        for (k, rot, _), lp in zip(jobs, lps):
            for r, v in zip(rot, lp):
                sc[k][r] += v
        for k, v in sc.items():
            self.cache[k] = max(k[1], key=lambda x: v[x])

    def chain(self, phs, t0):
        types = list(t0)
        out = [[0] * HMAX for _ in phs]
        for i in range(HMAX):
            keys = {}
            for b, ph in enumerate(phs):
                if i < len(ph):
                    keys[b] = (ph[i], tuple(allowed(types[b],
                                                    i == len(ph) - 1)))
            self.resolve(list(keys.values()))
            for b, k in keys.items():
                r = self.cache[k]
                out[b][i] = r
                types[b] = B.RNG[r]
        return out


def qwen_parses(qwen, items, W, widx):
    t0 = [W.type[widx[nm]] if nm in widx else None for nm, _ in items]
    phs = [ph[:HMAX] if ph else [""] for _, ph in items]
    rels = qwen.chain(phs, t0)
    return [(nm, r, len(p)) for (nm, _), r, p in zip(items, rels, phs)]


@torch.no_grad()
def answer(mem, addr, W, qs, parses, a, bs=128):
    recs = []
    for i in range(0, len(qs), bs):
        ch, pp = qs[i:i + bs], parses[i:i + bs]
        sa = torch.stack([addr.name(p[0]) for p in pp]).to(DEV)
        rels = torch.tensor([list(p[1]) for p in pp], device=DEV)
        hops = torch.tensor([max(1, min(HMAX, p[2])) for p in pp],
                            device=DEV)
        best, unk = B.beam_answer(mem, W, sa, rels, hops, a.K, a.tau)
        for q, p, b, u in zip(ch, pp, best.tolist(), unk.tolist()):
            h = len(q["rels"])
            if not q["gold"]:
                ok = bool(u)
            else:
                ok = (not u) and b in q["gold"]
            if ok:
                cause = "correct"
            elif p[0] != W.name[q["s"]]:
                cause = "name"
            elif p[2] != h:
                cause = "hop-count"
            elif list(p[1][:h]) != q["rels"]:
                cause = "relation"
            else:
                cause = "memory"
            recs.append(cause)
    return recs


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
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--zs_names", type=int, default=2)
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
    widx = {n: i for i, n in enumerate(W.name)}
    qsets = {}
    for c, ci in CONDS:
        pool = tr if c == "A" else te
        opens = B.TE_OPEN if c == "D" else B.TR_OPEN
        qsets[c] = [B2.qs_nested(W, 50 + 10 * ci + h, h, pool, opens,
                                 a.nq) for h in range(1, 5)]
    DW = B.World(64, random.Random(7))
    dummy = [[DW.name[s] for s in DW.groups[t][:a.zs_names]]
             for t in (0, 1)]
    qwen = QwenRel(lm, tok, dummy)
    acc = {v: {c: [] for c, _ in CONDS} for v in VARS}
    cause = {v: [collections.Counter() for _ in range(HMAX)]
             for v in VARS}
    seg = [[0, 0, 0] for _ in range(HMAX)]
    wrongs = collections.Counter()

    def keep(v, c, h, recs, per_h):
        per_h.append(recs.count("correct") / len(recs))
        if c == "B":
            cause[v][h].update(recs)

    for c, _ in CONDS:
        per_h = []
        for h in range(HMAX):
            qs = qsets[c][h]
            items = [(W.name[q["s"]], gold_phrases(q)) for q in qs]
            parses = qwen_parses(qwen, items, W, widx)
            keep("gold-seg+qwen", c, h,
                 answer(mem, addr, W, qs, parses, a), per_h)
            if c == "B":
                for q, (nm, ph), p in zip(qs, items, parses):
                    for i, r in enumerate(q["rels"]):
                        if p[1][i] != r:
                            wrongs[(ph[i], REL[r], REL[p[1][i]])] += 1
        acc["gold-seg+qwen"][c].append(per_h)
    log("ceiling (true phrases + Qwen) B: %s"
        % B2.fmt(acc["gold-seg+qwen"]["B"][-1]))

    for seed in range(a.seeds):
        torch.manual_seed(seed)
        m = B.Parser(hid, maxlen=96).to(DEV)
        log("seed %d: training slots parser" % seed)
        B2.train("slots", m, lm, tok, a, random.Random(seed + 1), tr)
        for c, _ in CONDS:
            ph_plain, ph_schema = [], []
            for h in range(HMAX):
                qs = qsets[c][h]
                plain, schema = slots_parses(m, lm, tok, qs, a.k_layers,
                                             W, widx)
                keep("slots", c, h, answer(mem, addr, W, qs, plain, a),
                     ph_plain)
                keep("slots+schema", c, h,
                     answer(mem, addr, W, qs, schema, a), ph_schema)
            acc["slots"][c].append(ph_plain)
            acc["slots+schema"][c].append(ph_schema)
        del m
        torch.manual_seed(seed)
        tg = BIOTagger(hid).to(DEV)
        log("seed %d: training tagger" % seed)
        train_tagger(tg, lm, tok, a, random.Random(seed + 1), tr)
        for c, _ in CONDS:
            per_h = []
            for h in range(HMAX):
                qs = qsets[c][h]
                items = parse_bio(tg, lm, tok, qs, a.k_layers)
                parses = qwen_parses(qwen, items, W, widx)
                keep("tagger+qwen", c, h,
                     answer(mem, addr, W, qs, parses, a), per_h)
                if c == "B":
                    for q, (nm, ph) in zip(qs, items):
                        gp = gold_phrases(q)
                        seg[h][0] += 1
                        seg[h][1] += int(ph == gp)
                        seg[h][2] += int(len(ph) == len(gp))
            acc["tagger+qwen"][c].append(per_h)
        del tg
        if DEV == "cuda":
            torch.cuda.empty_cache()
        log("seed %d B: slots %s | schema %s | tagger+qwen %s"
            % (seed, B2.fmt(acc["slots"]["B"][-1]),
               B2.fmt(acc["slots+schema"]["B"][-1]),
               B2.fmt(acc["tagger+qwen"]["B"][-1])))

    def mean(v):
        return sum(v) / len(v)

    print()
    print("=== Track B v6: let the frozen LM name the relations "
          "(%s, %d seeds) ===" % (a.lm, a.seeds))
    print("trained parts use v1 data only; Qwen made %d distinct "
          "phrase decisions (cached)" % len(qwen.cache))
    print("conditions: A seen phrases; B unseen phrase families; "
          "D unseen families + unseen openers")
    print()
    print("ANSWERS (mean over seeds; gold-seg+qwen needs no training)")
    print("%-14s %-5s %s" % ("variant", "cond",
                             "1-hop  2-hop  3-hop  4-hop   4-hop by seed"))
    for v in VARS:
        for c, _ in CONDS:
            rows = acc[v][c]
            hm = [mean([r[h] for r in rows]) for h in range(HMAX)]
            print("%-14s %-5s %s   %s"
                  % (v, c, B2.fmt(hm),
                     " ".join("%.3f" % r[3] for r in rows)))
    print()
    print("WHY CONDITION B FAILS (fraction of questions, seeds pooled)")
    print("%-14s hops  %s" % ("variant",
                              "  ".join("%9s" % c for c in CAUSES)))
    for v in VARS:
        for h in range(HMAX):
            cnt = cause[v][h]
            tot = max(sum(cnt.values()), 1)
            print("%-14s  %d    %s" % (v, h + 1, "  ".join(
                "%9.3f" % (cnt[c] / tot) for c in CAUSES)))
    print()
    print("TAGGER SPLITTING, condition B (seeds pooled)")
    print("hops  phrases exactly right  hop count right")
    for h in range(HMAX):
        n, ex, ho = seg[h]
        print("  %d          %5.3f            %5.3f"
              % (h + 1, ex / max(n, 1), ho / max(n, 1)))
    print()
    print("QWEN MISTAKES on true phrases, condition B "
          "(phrase: true -> picked, count)")
    for (ph, t, p), n in wrongs.most_common(12):
        print("  %-40s %s -> %s  %d" % (ph, t, p, n))
    if not wrongs:
        print("  none")
    log("done")


if __name__ == "__main__":
    main()
