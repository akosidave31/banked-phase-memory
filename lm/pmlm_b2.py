"""
pmlm_b2.py - Track B v2: a dedicated relation model trained on the
paraphrase dataset (relations_v1.json), tested on held-out families.

Split: for each relation, phrase families are sorted by size; the
families at sorted positions 1, 4, 7, ... are HELD OUT (test only).
Questions: nested noun phrases "What is NP_h ... NP_1 NAME?"
(1-4 hops) and natural 1-hop questions ("Who employs NAME?").
Two parsers on the same frozen SmolLM2 features and the same data:
  slots  : Track B v1 parser (name start/end, slot attention, hops)
  tagger : relation model - tags each token as filler / name /
           phrase-of-hop-i; hop count = number of phrases; each
           phrase is pooled and matched to relation prototypes
Both drive the same banked phase memory (never trained).
Needs pmlm_b.py and relations_v1.json in the same folder.
"""
import argparse
import collections
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pmlm_b as B

T0 = time.time()
DEV = B.DEV
NR, HMAX = B.NR, B.HMAX
REL_NAMES = ["works_at", "lives_in", "located_in", "friend", "ceo"]
NTAG = 2 + HMAX


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def load_split(path):
    data = json.load(open(path))
    tr = {"np": [[] for _ in range(NR)], "q": [[] for _ in range(NR)]}
    te = {"np": [[] for _ in range(NR)], "q": [[] for _ in range(NR)]}
    held = {}
    for r, name in enumerate(REL_NAMES):
        d = data[name]
        cnt = collections.Counter()
        for kind in ("noun_phrases", "questions", "facts"):
            for it in d[kind]:
                cnt[it["family"]] += 1
        fams = sorted(cnt, key=lambda f: (-cnt[f], f))
        test_f = {f for i, f in enumerate(fams) if i % 3 == 1}
        held[name] = sorted(test_f)
        for kind, key in (("noun_phrases", "np"), ("questions", "q")):
            for it in d[kind]:
                t = it["text"].strip()
                if key == "np" and "{X}" in t:
                    continue
                if key == "q" and t.count("{X}") != 1:
                    continue
                dst = te if it["family"] in test_f else tr
                dst[key][r].append(t)
    for pool, nm in ((tr, "train"), (te, "test")):
        for r in range(NR):
            assert pool["np"][r], "no %s phrases for %s" % (nm, REL_NAMES[r])
    return tr, te, held


def path(world, rng, h):
    t = rng.choice([0, 1])
    s = rng.choice(world.groups[t])
    rels = []
    for i in range(h):
        opts = B.OUT if i == h - 1 else B.CONT
        r = rng.choice(opts[t])
        rels.append(r)
        t = B.RNG[r]
    return s, rels


def nested_q(world, rng, h, pool, opens):
    s, rels = path(world, rng, h)
    text, segs = rng.choice(opens), []
    for k, r in enumerate(reversed(rels)):
        text += " "
        a = len(text)
        text += rng.choice(pool["np"][r])
        segs.append((a, len(text), h - k))
    text += " "
    n0 = len(text)
    text += world.name[s]
    n1 = len(text)
    text += "?"
    return dict(text=text, name=(n0, n1), segs=segs, s=s, rels=rels,
                gold=B.follow(world, s, rels))


def natural_q(world, rng, pool):
    rs = [r for r in range(NR) if pool["q"][r]]
    r = rng.choice(rs)
    s = rng.choice(world.groups[B.DOM[r]])
    tm = rng.choice(pool["q"][r])
    i = tm.index("{X}")
    nm = world.name[s]
    text = tm[:i] + nm + tm[i + 3:]
    return dict(text=text, name=(i, i + len(nm)),
                segs=[(0, len(text), 1)], s=s, rels=[r],
                gold=B.follow(world, s, [r]))


def train_q(world, rng, pool):
    if rng.random() < 0.7:
        return nested_q(world, rng, rng.randint(1, HMAX), pool,
                        B.TR_OPEN)
    return natural_q(world, rng, pool)


def qs_nested(world, seed, h, pool, opens, n):
    rng = random.Random(seed)
    return [nested_q(world, rng, h, pool, opens) for _ in range(n)]


def tag_targets(enc, qs):
    T = enc["input_ids"].shape[1]
    tags = torch.zeros(len(qs), T, dtype=torch.long)
    for b, q in enumerate(qs):
        offs = enc["offset_mapping"][b].tolist()
        for j, (x, y) in enumerate(offs):
            if y <= x:
                continue
            if not any(c.isalnum() for c in q["text"][x:y]):
                continue
            if x < q["name"][1] and y > q["name"][0]:
                tags[b, j] = 1
                continue
            for a, e, hop in q["segs"]:
                if x < e and y > a:
                    tags[b, j] = 1 + hop
                    break
    return tags


def clean(s):
    s = s.strip(" ?.,")
    if s.endswith("'s"):
        s = s[:-2]
    return s.strip()


@torch.no_grad()
def feats(lm, enc, k):
    out = lm.model(input_ids=enc["input_ids"].to(DEV),
                   attention_mask=enc["attention_mask"].to(DEV),
                   output_hidden_states=True)
    return torch.stack(out.hidden_states[-k:]).float().mean(0)


class RelTagger(nn.Module):
    def __init__(self, hid, d=256, layers=2, maxlen=96):
        super().__init__()
        self.proj = nn.Linear(hid, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(
            d, 4, 4 * d, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers,
                                         enable_nested_tensor=False)
        self.tag = nn.Linear(d, NTAG)
        self.emb = nn.Linear(d, d)
        self.proto = nn.Parameter(torch.randn(NR, d) * 0.1)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, f, mask):
        T = f.shape[1]
        h = self.proj(f) + self.pos.weight[None, :T]
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.tag(h), self.emb(h)

    def rel_logits(self, e, sel):
        den = sel.sum(-1, keepdim=True).clamp(min=1)
        pooled = torch.einsum("bht,btd->bhd", sel, e) / den
        z = F.normalize(pooled, dim=-1)
        p = F.normalize(self.proto, dim=-1)
        return self.scale * z @ p.T


def train(kind, m, lm, tok, a, rng, pool):
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    every = max(a.steps // 6, 1)
    hop_ids = torch.arange(HMAX, device=DEV)
    world = None
    for step in range(1, a.steps + 1):
        if step == 1 or step % 50 == 0:
            world = B.World(a.n_train, rng)
        qs = [train_q(world, rng, pool) for _ in range(a.bs)]
        enc = B.tokenize(tok, qs)
        mask = enc["attention_mask"].bool().to(DEV)
        f = feats(lm, enc, a.k_layers)
        rel_t = torch.full((len(qs), HMAX), -100, dtype=torch.long)
        for i, q in enumerate(qs):
            rel_t[i, :len(q["rels"])] = torch.tensor(q["rels"])
        rel_t = rel_t.to(DEV)
        hop_t = torch.tensor([len(q["rels"]) - 1 for q in qs],
                             device=DEV)
        if kind == "slots":
            for q in qs:
                q["span"] = q["name"]
            st_t, en_t = B.gold_spans(enc, qs)
            st, en, rl, hp = m(f, mask)
            loss = (F.cross_entropy(st, st_t)
                    + F.cross_entropy(en, en_t)
                    + F.cross_entropy(rl.reshape(-1, NR),
                                      rel_t.reshape(-1),
                                      ignore_index=-100)
                    + F.cross_entropy(hp, hop_t))
            acc = ((st.argmax(-1) == st_t)
                   & (en.argmax(-1) == en_t)).float().mean().item()
        else:
            tags = tag_targets(enc, qs).to(DEV).masked_fill(~mask, -100)
            tl, e = m(f, mask)
            sel = (tags[:, None, :]
                   == (2 + hop_ids)[None, :, None]).float()
            rl = m.rel_logits(e, sel)
            loss = (F.cross_entropy(tl.reshape(-1, NTAG),
                                    tags.reshape(-1), ignore_index=-100)
                    + F.cross_entropy(rl.reshape(-1, NR),
                                      rel_t.reshape(-1),
                                      ignore_index=-100))
            ok = (tl.argmax(-1) == tags)[mask]
            acc = ok.float().mean().item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % every == 0:
            log("  %s step %d loss=%.3f acc=%.3f"
                % (kind, step, loss.item(), acc))


@torch.no_grad()
def parse_slots(m, lm, tok, qs, k):
    m.eval()
    enc = B.tokenize(tok, qs)
    mask = enc["attention_mask"].bool().to(DEV)
    st, en, rl, hp = m(feats(lm, enc, k), mask)
    T = st.shape[1]
    i = torch.arange(T, device=DEV)
    valid = (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] <= 12)
    sc = (st[:, :, None] + en[:, None, :]).masked_fill(~valid[None],
                                                       -1e9)
    flat = sc.reshape(len(qs), -1).argmax(-1)
    s_i, e_i = (flat // T).tolist(), (flat % T).tolist()
    offs = enc["offset_mapping"]
    names = []
    for b, q in enumerate(qs):
        x0, x1 = int(offs[b, s_i[b], 0]), int(offs[b, e_i[b], 1])
        names.append(clean(q["text"][x0:x1]))
    m.train()
    return names, rl.argmax(-1), hp.argmax(-1) + 1


@torch.no_grad()
def parse_tagger(m, lm, tok, qs, k):
    m.eval()
    enc = B.tokenize(tok, qs)
    mask = enc["attention_mask"].bool().to(DEV)
    tl, e = m(feats(lm, enc, k), mask)
    pt = tl.argmax(-1).masked_fill(~mask, 0)
    hop_ids = torch.arange(HMAX, device=DEV)
    sel = (pt[:, None, :] == (2 + hop_ids)[None, :, None]).float()
    rels = m.rel_logits(e, sel).argmax(-1)
    present = (sel.sum(-1) > 0).long()
    hops = (present * (hop_ids + 1)[None]).amax(-1).clamp(min=1)
    offs = enc["offset_mapping"]
    ptl = pt.tolist()
    names = []
    for b, q in enumerate(qs):
        idx = [j for j, v in enumerate(ptl[b]) if v == 1]
        if idx:
            x0, x1 = int(offs[b, idx[0], 0]), int(offs[b, idx[-1], 1])
            names.append(clean(q["text"][x0:x1]))
        else:
            names.append("")
    m.train()
    return names, rels, hops


def diag(fn, m, lm, tok, world, qs, k, bs=64):
    n_ok = h_ok = 0
    r_ok, r_n = [0] * HMAX, [0] * HMAX
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        names, rels, hops = fn(m, lm, tok, ch, k)
        for q, nm, rr, hh in zip(ch, names, rels.tolist(),
                                 hops.tolist()):
            n_ok += int(nm == world.name[q["s"]])
            h_ok += int(hh == len(q["rels"]))
            for j, r in enumerate(q["rels"]):
                r_n[j] += 1
                r_ok[j] += int(rr[j] == r)
    return (n_ok / len(qs), h_ok / len(qs),
            [o / max(c, 1) for o, c in zip(r_ok, r_n)])


def answer_acc(fn, m, lm, tok, mem, addr, world, qs, a, bs=64):
    ok = 0
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        names, rels, hops = fn(m, lm, tok, ch, a.k_layers)
        sa = torch.stack([addr.name(nm) for nm in names]).to(DEV)
        best, unk = B.beam_answer(mem, world, sa, rels, hops, a.K, a.tau)
        for q, b, u in zip(ch, best.tolist(), unk.tolist()):
            if not q["gold"]:
                ok += int(u)
            else:
                ok += int((not u) and b in q["gold"])
    return ok / len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="HuggingFaceTB/SmolLM2-135M-Instruct")
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
    ap.add_argument("--big", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tr, te, held = load_split(a.data)
    for name in REL_NAMES:
        log("held-out families %-10s: %s" % (name, ", ".join(held[name])))
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.lm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.lm, dtype=dtype)
    lm = lm.to(DEV).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    hid = lm.config.hidden_size
    models = {}
    for kind in ("slots", "tagger"):
        rng = random.Random(a.seed + 1)
        m = (B.Parser(hid, maxlen=96) if kind == "slots"
             else RelTagger(hid)).to(DEV)
        npar = sum(p.numel() for p in m.parameters())
        log("training %s parser (%d params)" % (kind, npar))
        train(kind, m, lm, tok, a, rng, tr)
        models[kind] = (m, npar)

    addr = B.Addr(a.D)
    mem = B.Memory(a.D, a.L, a.seed)
    W = B.World(a.n_train, random.Random(2000))
    B.prepare(W, addr)
    mem.write(W)
    seen = [qs_nested(W, 10 + h, h, tr, B.TR_OPEN, a.nq)
            for h in range(1, 5)]
    unseen = [qs_nested(W, 20 + h, h, te, B.TE_OPEN, a.nq)
              for h in range(1, 5)]
    rn = random.Random(30)
    nat = [natural_q(W, rn, te) for _ in range(a.nq)]
    fns = {"slots": parse_slots, "tagger": parse_tagger}
    res = {}
    for kind in ("slots", "tagger"):
        m, fn = models[kind][0], fns[kind]
        dg = diag(fn, m, lm, tok, W, [q for hs in unseen for q in hs],
                  a.k_layers)
        s_acc = [answer_acc(fn, m, lm, tok, mem, addr, W, q, a)
                 for q in seen]
        u_acc = [answer_acc(fn, m, lm, tok, mem, addr, W, q, a)
                 for q in unseen]
        n_acc = answer_acc(fn, m, lm, tok, mem, addr, W, nat, a)
        res[kind] = (dg, s_acc, u_acc, n_acc)
        log("evaluated %s" % kind)
    Wb = B.World(a.big, random.Random(3000))
    B.prepare(Wb, addr)
    mem.write(Wb)
    big_q = [qs_nested(Wb, 40 + h, h, te, B.TE_OPEN, a.nq)
             for h in range(1, 5)]
    tm = models["tagger"][0]
    big = [answer_acc(parse_tagger, tm, lm, tok, mem, addr, Wb, q, a)
           for q in big_q]
    log("evaluated %d-entity world" % a.big)

    print()
    print("=== Track B v2: relation model vs v1 parser, same data ===")
    print("slots parser %d params; tagger (relation model) %d params"
          % (models["slots"][1], models["tagger"][1]))
    print()
    print("PARSER on UNSEEN families     name   hops   rel h1  h2"
          "     h3     h4")
    for kind in ("slots", "tagger"):
        dg = res[kind][0]
        print("%-8s                     %5.3f  %5.3f  %s"
              % (kind, dg[0], dg[1], fmt(dg[2])))
    print()
    print("ANSWERS via memory, new %d-entity world" % a.n_train)
    print("parser   families   1-hop  2-hop  3-hop  4-hop")
    for kind in ("slots", "tagger"):
        _, s_acc, u_acc, _ = res[kind]
        print("%-8s seen       %s" % (kind, fmt(s_acc)))
        print("%-8s UNSEEN     %s" % (kind, fmt(u_acc)))
    print()
    print("natural 1-hop questions, UNSEEN families:  slots %.3f   "
          "tagger %.3f" % (res["slots"][3], res["tagger"][3]))
    print()
    print("%d entities, tagger, UNSEEN families:  %s"
          % (a.big, fmt(big)))
    print()
    print("Track B v1 reference (2 phrasings per relation, unseen "
          "words): 0.650  0.793  0.527  0.560")
    log("done")


def fmt(v):
    return "  ".join("%5.3f" % x for x in v)


if __name__ == "__main__":
    main()
