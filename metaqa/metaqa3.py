"""
metaqa3.py - MetaQA stage 2: the full system, question in -> answer out.
  topic entity : the bracketed name (MetaQA's standard protocol)
  path parser  : frozen Qwen 0.5B features + small trained parser
                 (hop count, relation per hop, topic mode); relations
                 decoded under the schema (film -> person/attribute by a
                 forward relation, back to films by a reverse one)
  memory       : banked phase memory from stage 1b (spread hubs, one
                 bank per key slot), never trained
  reasoner     : beam (K=20); the mode says how the topic is excluded
Training labels come from training questions only: each training
template's path is recovered from its question/answer pairs (no hand
labels). Test questions get labels from the same template map only to
measure parser accuracy. Needs metaqa1.py, metaqa2.py and the MetaQA
2-hop / 3-hop qa_train.txt and qa_test.txt files.
"""
import argparse
import collections
import os
import random
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import metaqa1 as M
import metaqa2 as M2

T0 = time.time()
DEV = M.DEV
HM = 3


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def tmpl(q):
    return re.sub(r"\[.+?\]", "[E]", q)


def load_split(root, hop, split, look):
    d = os.path.join(root, "%d-hop" % hop)
    for p in (os.path.join(d, "vanilla", "qa_%s.txt" % split),
              os.path.join(d, "qa_%s.txt" % split)):
        if os.path.exists(p):
            break
    else:
        return None
    out = []
    for line in open(p, encoding="utf-8"):
        q, _, ans = line.rstrip("\n").partition("\t")
        m = re.search(r"\[(.+?)\]", q)
        tid = look(m.group(1)) if m else None
        gold = {look(x) for x in ans.split("|")} - {None}
        if not gold:
            tid = None
        out.append((q, tid, gold, tmpl(q)))
    return out


@torch.no_grad()
def feats(lm, enc, k):
    out = lm.model(input_ids=enc["input_ids"].to(DEV),
                   attention_mask=enc["attention_mask"].to(DEV),
                   output_hidden_states=True)
    return torch.stack(out.hidden_states[-k:]).float().mean(0)


class PathParser(nn.Module):
    def __init__(self, hid, nrel, d=256, layers=2, maxlen=128):
        super().__init__()
        self.proj = nn.Linear(hid, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(
            d, 4, 4 * d, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers,
                                         enable_nested_tensor=False)
        self.slots = nn.Parameter(torch.randn(HM, d) * 0.02)
        self.att = nn.MultiheadAttention(d, 4, batch_first=True)
        self.rel = nn.Linear(d, nrel)
        self.hops = nn.Linear(d, HM)
        self.mode = nn.Linear(d, 3)

    def forward(self, f, mask):
        b, T, _ = f.shape
        h = self.proj(f) + self.pos.weight[None, :T]
        h = self.enc(h, src_key_padding_mask=~mask)
        q = self.slots[None].expand(b, -1, -1)
        o, _ = self.att(q, h, h, key_padding_mask=~mask)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.rel(o), self.hops(pooled), self.mode(pooled)


def encode(tok, texts):
    tok.padding_side = "right"
    return tok(texts, padding=True, truncation=True, max_length=128,
               return_tensors="pt")


def train_parser(m, lm, tok, train, a, rng, nrel):
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    every = max(a.steps // 6, 1)
    for step in range(1, a.steps + 1):
        batch = [train[rng.randrange(len(train))] for _ in range(a.bs)]
        enc = encode(tok, [b[0] for b in batch])
        mask = enc["attention_mask"].bool().to(DEV)
        rl, hp, md = m(feats(lm, enc, a.k_layers), mask)
        rel_t = torch.full((len(batch), HM), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            rel_t[i, :len(b[1])] = torch.tensor(b[1])
        rel_t = rel_t.to(DEV)
        hop_t = torch.tensor([len(b[1]) - 1 for b in batch], device=DEV)
        mode_t = torch.tensor([b[2] for b in batch], device=DEV)
        loss = (F.cross_entropy(rl.reshape(-1, nrel), rel_t.reshape(-1),
                                ignore_index=-100)
                + F.cross_entropy(hp, hop_t) + F.cross_entropy(md, mode_t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % every == 0:
            ok = ((hp.argmax(-1) == hop_t) & (md.argmax(-1) == mode_t)
                  & ((rl.argmax(-1) == rel_t) | (rel_t == -100)).all(-1))
            log("  parser step %d loss=%.3f path acc=%.3f"
                % (step, loss.item(), ok.float().mean().item()))


@torch.no_grad()
def parse(m, lm, tok, items, a, is_movie, NB, bs=128):
    m.eval()
    out = []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        enc = encode(tok, [x[0] for x in ch])
        mask = enc["attention_mask"].bool().to(DEV)
        rl, hp, md = m(feats(lm, enc, a.k_layers), mask)
        rl = rl.float().cpu().tolist()
        hops = (hp.argmax(-1) + 1).tolist()
        modes = md.argmax(-1).tolist()
        for j, x in enumerate(ch):
            movie = x[1] in is_movie if x[1] is not None else True
            path = []
            for s in range(hops[j]):
                row = rl[j][s]
                cand = range(NB) if movie else range(NB, 2 * NB)
                path.append(max(cand, key=lambda r: row[r]))
                movie = not movie
            out.append((tuple(path), modes[j]))
    m.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/mqa")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B-Instruct")
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
    ap.add_argument("--seeds", type=int, default=3)
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
    is_movie = {e for e, rs in outr.items() if any(r < NB for r in rs)}
    data = {}
    for hop in (1, 2, 3):
        tr = load_split(a.root, hop, "train", look)
        te = load_split(a.root, hop, "test", look)
        if tr and te:
            data[hop] = (tr, te)
    hops = sorted(data)
    log("KB %d entities; hops with train+test: %s; train %s; test %s"
        % (len(names), hops, [len(data[h][0]) for h in hops],
           [len(data[h][1]) for h in hops]))

    tpath, tinfo = {}, {}
    for hop in hops:
        groups = collections.defaultdict(list)
        for q, tid, gold, t in data[hop][0]:
            if tid is not None:
                groups[t].append((tid, gold))
        ns = a.sample2 if hop <= 2 else a.sample3
        fs = []
        for t, items in groups.items():
            smp = items[:ns]
            sc = M2.recover2(adj, outr, smp, hop)
            if sc:
                best = max(sc, key=sc.get)
                tpath[(hop, t)] = best
                fs.append(sc[best] / len(smp))
        tinfo[hop] = (len(groups), sum(fs) / max(len(fs), 1))
        log("%d-hop: %d training templates, mean path F1 %.3f"
            % (hop, tinfo[hop][0], tinfo[hop][1]))
    train = []
    for hop in hops:
        for q, tid, gold, t in data[hop][0]:
            lab = tpath.get((hop, t))
            if lab and lab[0]:
                train.append((q, list(lab[0]), int(lab[1])))
    log("training questions used: %d" % len(train))

    sp = M2.SpreadMem(len(names), NREL, a.D, a.seed)
    sp.write_spread(torch.tensor(S), torch.tensor(Rr), torch.tensor(O),
                    a.L, len(names), NREL)
    log("memory built: %d banks, fullest %d" % (sp.nb, max(sp.load)))

    dtype = torch.float16 if DEV == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.lm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.lm, dtype=dtype)
    lm = lm.to(DEV).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    hid = lm.config.hidden_size
    log("LM %s loaded" % a.lm)

    cache = {}
    res = {h: [] for h in hops}
    confuse = collections.Counter()
    unseen_t = {h: sum(1 for x in data[h][1] if (h, x[3]) not in tpath)
                for h in hops}
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        m = PathParser(hid, NREL).to(DEV)
        log("seed %d: training parser" % seed)
        train_parser(m, lm, tok, train, a, random.Random(seed + 1), NREL)
        for hop in hops:
            te = data[hop][1]
            preds = parse(m, lm, tok, te, a, is_movie, NB)
            need = collections.defaultdict(list)
            for x, (path, mode) in zip(te, preds):
                key = (x[1], path, mode)
                if x[1] is not None and key not in cache:
                    need[(path, mode)].append(x[1])
            for (path, mode), tids in need.items():
                tids = sorted(set(tids))
                bs = max(8, 2560 // a.K)
                for i in range(0, len(tids), bs):
                    ch = tids[i:i + bs]
                    top = torch.tensor(ch, device=DEV)
                    best = M2.answer(sp, top, list(path), a.K, mode, True,
                                     a.cap).tolist()
                    for t_, b_ in zip(ch, best):
                        cache[(t_, path, mode)] = b_
            n = len(te)
            pok = hit = hit_ok = hit_bad = n_ok = 0
            for x, (path, mode) in zip(te, preds):
                lab = tpath.get((hop, x[3]))
                good = bool(lab) and lab[0] == path and int(lab[1]) == mode
                h = (x[1] is not None
                     and cache.get((x[1], path, mode)) in x[2])
                pok += good
                hit += h
                if good:
                    n_ok += 1
                    hit_ok += h
                else:
                    hit_bad += h
                    if lab and seed == 0:
                        g = " > ".join(rname(r) for r in lab[0]) + \
                            " [%s]" % "nfa"[int(lab[1])]
                        p = " > ".join(rname(r) for r in path) + \
                            " [%s]" % "nfa"[mode]
                        confuse[(hop, g, p)] += 1
            res[hop].append((pok / n, hit / n, hit_ok / max(n_ok, 1),
                             hit_bad / max(n - n_ok, 1), n - n_ok))
            log("seed %d %d-hop: path acc %.4f  Hits@1 %.4f"
                % (seed, hop, pok / n, hit / n))
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()

    def mean(v):
        return sum(v) / len(v)

    print()
    print("=== MetaQA stage 2: full system, question in -> answer out "
          "(%s parser + phase memory, K=%d, %d seeds) ==="
          % (a.lm, a.K, a.seeds))
    for hop in hops:
        print("%d-hop: %d training templates (path F1 %.3f); test "
              "questions whose template is not in training: %d"
              % (hop, tinfo[hop][0], tinfo[hop][1], unseen_t[hop]))
    print()
    print("RESULTS (mean over seeds)")
    print("hop      n  path acc   Hits@1   Hits@1|path ok  Hits@1|path "
          "wrong  wrong paths")
    for hop in hops:
        r = res[hop]
        print("%d   %6d   %6.4f   %6.4f        %6.4f          %6.4f     "
              "%6.0f" % (hop, len(data[hop][1]), mean([x[0] for x in r]),
                         mean([x[1] for x in r]), mean([x[2] for x in r]),
                         mean([x[3] for x in r]), mean([x[4] for x in r])))
    print("Hits@1 by seed: " + " | ".join(
        "%d-hop %s" % (h, " ".join("%.4f" % x[1] for x in res[h]))
        for h in hops))
    print("reference, memory with the true path (stage 1b): 2-hop 0.9844, "
          "3-hop 0.9928")
    print()
    print("MOST COMMON PATH MISTAKES (seed 0; true -> predicted; [n/f/a] "
          "= topic mode)")
    for (hop, g, p), c in confuse.most_common(10):
        print("  %d  %-30s -> %-30s %d" % (hop, g, p, c))
    if not confuse:
        print("  none")
    log("done")


if __name__ == "__main__":
    main()
