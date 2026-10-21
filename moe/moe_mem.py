"""
moe_mem.py - Small MoE, step 2: a memory expert that reads the frozen
phase memory.
Starts from the step-1 MoE checkpoint and continues two models on the same
data (half FineWeb-Edu text, half fact sentences from the MetaQA movie
knowledge base, e.g. "<Titanic> was directed by <James Cameron>."):
  baseline : the MoE alone; facts must be learned into its weights
  memory   : the same MoE plus a memory expert. Where an entity comes next,
             a small controller points at the subject among recent entities
             and picks the top-2 relations, reads the frozen banked phase
             memory, and votes on the answer; a learned gate mixes this vote
             with the MoE's own.
Entities are single tokens whose input and output vectors are the memory's
own random phase vectors through shared projections, so neither model holds
per-entity weights. Every fact is in the memory; 10% of (film, relation)
keys never appear in training text. The controller is trained with labels
from the generated sentences (relation and subject).
Tests: facts seen in text, facts only in memory, memory blanked, and
in-place edits with no retraining.
Needs moe_train.py, metaqa1.py ... metaqa4.py, the step-1 data and MoE
checkpoint, and MetaQA kb.txt.
"""
import argparse
import collections
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import moe_train as MT
import metaqa1 as M
import metaqa4 as M4

T0 = time.time()
DEV = MT.DEV
AMP = DEV == "cuda"
HEAD = ["directed_by", "written_by", "starred_actors", "release_year"]
TPL = {
    "directed_by": (["{S} was directed by {O}.", "The director of {S} is {O}."],
                    ["{S} directed {O}.", "{S} is the director of {O}."]),
    "written_by": (["{S} was written by {O}.", "The writer of {S} is {O}."],
                   ["{S} wrote {O}.", "{S} is a writer of {O}."]),
    "starred_actors": (["{S} stars {O}.", "One actor in {S} is {O}."],
                       ["{S} acted in {O}.", "{S} appears in {O}."]),
    "release_year": (["{S} was released in {O}.", "{S} came out in {O}."],
                     ["A film released in {S} is {O}.",
                      "In {S} the film released was {O}."]),
    "in_language": (["{S} is in {O}.", "The language of {S} is {O}."],
                    ["A film in {S} is {O}.", "{S} is the language of {O}."]),
    "has_genre": (["The genre of {S} is {O}.",
                   "{S} belongs to the genre {O}."],
                  ["One {S} film is {O}.", "An example of {S} is {O}."]),
    "has_tags": (["{S} is tagged {O}.", "A tag of {S} is {O}."],
                 ["A film tagged {S} is {O}.", "{S} is a tag of {O}."]),
    "has_imdb_votes": (["The vote level of {S} is {O}.",
                        "{S} has votes {O}."],
                       ["A film with {S} votes is {O}.",
                        "With {S} votes there is {O}."]),
    "has_imdb_rating": (["The rating of {S} is {O}.", "{S} is rated {O}."],
                        ["A film rated {S} is {O}.",
                         "With rating {S} there is {O}."]),
}


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def tpl_for(rn):
    if rn in TPL:
        return TPL[rn]
    w = rn.replace("_", " ")
    return (["{S} has %s {O}." % w], ["{S} is the %s of {O}." % w])


def split_tok(tok, t):
    i, j = t.index("{S}"), t.index("{O}")
    pre, mid, suf = t[:i].strip(), t[i + 3:j].rstrip(), t[j + 3:]
    assert mid.strip(), t
    pre_ids = tok.encode(" " + pre).ids if pre else []
    mid_ids = tok.encode(mid if mid.startswith(" ") else " " + mid).ids
    suf_ids = tok.encode(suf).ids if suf else []
    return pre_ids, mid_ids, suf_ids


class Facts:
    def __init__(self, tok, rels, trip, NB, V, hold, seed):
        self.V, self.NB = V, NB
        self.tpl = []
        for rr in range(2 * NB):
            fwd, rev = tpl_for(rels[rr % NB])
            self.tpl.append([split_tok(tok, t)
                             for t in (fwd if rr < NB else rev)])
        self.obj = collections.defaultdict(set)
        keys = collections.defaultdict(list)
        for s, r, o in trip:
            self.obj[(s, r)].add(o)
            self.obj[(o, r + NB)].add(s)
            keys[(s, r)].append(o)
        allk = sorted(keys)
        rng = random.Random(seed)
        self.held = set(rng.sample(allk, int(hold * len(allk))))
        self.held_list = sorted(self.held)
        self.train_keys = [k for k in allk if k not in self.held]
        self.n_keys = len(allk)
        self.n_held_facts = sum(len(keys[k]) for k in self.held)
        fwd, rev = [], []
        for (s, r), objs in keys.items():
            if (s, r) in self.held:
                continue
            for o in objs:
                fwd.append((s, r, o))
                rev.append((o, r + NB, s))
        self.fwd, self.rev = np.array(fwd), np.array(rev)

    def seq(self, rng, L):
        V = self.V
        toks, ans, last_ent = [], [], -1
        while len(toks) < L + 1:
            src = self.fwd if rng.random() < 0.7 else self.rev
            s, rr, o = src[rng.integers(len(src))]
            tp = self.tpl[rr]
            pre, mid, suf = tp[rng.integers(len(tp))]
            toks.extend(pre)
            sp = len(toks)
            toks.append(V + int(s))
            toks.extend(mid)
            t = len(toks) - 1
            toks.append(V + int(o))
            if t + 1 <= L:
                ans.append((t, sp, last_ent, int(rr), int(o)))
            last_ent = len(toks) - 1
            toks.extend(suf)
        return toks[:L + 1], ans


class Reader:
    """Reads (subject, relation) keys from the banked phase memory on the
    GPU: every slot of the key (up to cap) is unbound and summed."""

    def __init__(self, mem, nrel, L, cap):
        self.mem, self.nrel, self.L, self.cap = mem, nrel, L, cap
        self.SM = 1 << 13
        self.refresh()

    def refresh(self):
        items = list(self.mem.unit.items())
        codes = torch.tensor([(s * self.nrel + r) * self.SM + j
                              for (s, r, j), _ in items], dtype=torch.long)
        banks = torch.tensor([b for _, b in items], dtype=torch.long)
        o = codes.argsort()
        self.codes, self.banks = codes[o].to(DEV), banks[o].to(DEV)

    @torch.no_grad()
    def read(self, s, r, blank=False, chunk=16384):
        n, D = s.shape[0], self.mem.D
        re_ = torch.zeros(n, D, device=DEV)
        im = torch.zeros(n, D, device=DEV)
        if blank or n == 0:
            return torch.cat([re_, im], -1)
        ns = ((self.mem.deg[s, r] + self.L - 1) // self.L).clamp(max=self.cap)
        rows = torch.repeat_interleave(torch.arange(n, device=DEV), ns)
        if rows.numel() == 0:
            return torch.cat([re_, im], -1)
        starts = torch.cumsum(ns, 0) - ns
        js = (torch.arange(rows.numel(), device=DEV)
              - torch.repeat_interleave(starts, ns))
        for i in range(0, rows.numel(), chunk):
            rc, jc = rows[i:i + chunk], js[i:i + chunk]
            sc, rr = s[rc], r[rc]
            code = (sc * self.nrel + rr) * self.SM + jc
            ix = torch.searchsorted(self.codes, code).clamp(
                max=self.codes.numel() - 1)
            ok = (self.codes[ix] == code).float()[:, None]
            key = self.mem.addr[sc] * self.mem.R[rr] * self.mem.Sv[jc]
            v = torch.roll(self.mem.B[self.banks[ix]] * key.conj(), -1, -1)
            re_.index_add_(0, rc, v.real.contiguous() * ok)
            im.index_add_(0, rc, v.imag.contiguous() * ok)
        return torch.cat([re_, im], -1)


class MemLM(nn.Module):
    def __init__(self, a, Afeat, memory):
        super().__init__()
        self.lm = MT.LM(a, True)
        self.V = a.vocab
        self.register_buffer("Afeat", Afeat, persistent=False)
        F2 = Afeat.shape[1]
        self.D = F2 // 2
        self.ein = nn.Linear(F2, a.d, bias=False)
        nn.init.normal_(self.ein.weight, 0.0, 0.02 / math.sqrt(self.D))
        self.eout = nn.Linear(a.d, F2, bias=False)
        nn.init.normal_(self.eout.weight, 0.0, 0.02)
        self.isent = nn.Linear(a.d, 1)
        self.memory = memory
        if memory:
            self.gate = nn.Linear(a.d, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            self.rel = nn.Linear(a.d, a.nrel)
            self.pq = nn.Linear(a.d, 64, bias=False)
            self.pk = nn.Linear(a.d, 64, bias=False)
            self.log_beta = nn.Parameter(torch.tensor(math.log(20.0)))

    def hidden(self, idx):
        V = self.V
        ent = idx >= V
        x = self.lm.emb(idx.clamp(max=V - 1))
        if bool(ent.any()):
            ex = torch.zeros_like(x)
            ex[ent] = self.ein(self.Afeat[idx[ent] - V].float()).to(x.dtype)
            x = torch.where(ent[..., None], ex, x)
        aux = z = 0.0
        for b in self.lm.blocks:
            x, a_, z_ = b(x, self.lm.cos, self.lm.sin)
            aux = aux + a_
            z = z + z_
        return self.lm.nf(x), aux, z

    def ent_logits(self, ha):
        q = self.eout(ha).to(self.Afeat.dtype)
        return (q @ self.Afeat.T).float() / math.sqrt(self.D)

    def answer(self, h, ab, at, cpos, x, reader, blank=False):
        ha = h[ab, at]
        lpm = F.log_softmax(self.ent_logits(ha), -1)
        if not self.memory:
            return lpm, {}
        n = ha.shape[0]
        valid = cpos >= 0
        cp = cpos.clamp(min=0)
        hk = h[ab[:, None], cp]
        sc = (self.pq(ha)[:, None, :] * self.pk(hk)).sum(-1).float() / 8.0
        sc = sc.masked_fill(~valid, -1e4)
        att = sc.softmax(-1)
        rl = self.rel(ha).float()
        pv, pi = rl.softmax(-1).topk(2, -1)
        ents = (x[ab[:, None], cp] - self.V).clamp(min=0)
        s_flat = ents[:, :, None].expand(n, 2, 2).reshape(-1)
        r_flat = pi[:, None, :].expand(n, 2, 2).reshape(-1)
        feat = reader.read(s_flat, r_flat, blank).view(n, 4, -1)
        w = (att[:, :, None] * pv[:, None, :]).reshape(n, 4)
        q = (w[..., None] * feat).sum(1)
        smem = (q.to(self.Afeat.dtype) @ self.Afeat.T).float() / self.D
        lpmem = F.log_softmax(smem * self.log_beta.exp(), -1)
        g = self.gate(ha).float().squeeze(-1)
        lp = torch.logsumexp(torch.stack(
            [F.logsigmoid(g)[:, None] + lpmem,
             F.logsigmoid(-g)[:, None] + lpm]), 0)
        return lp, {"rel": rl, "ptr": sc, "gate": torch.sigmoid(g)}


def lr_at(a, s, steps):
    warm = max(1, int(0.02 * steps))
    if s < warm:
        return a.lr * (s + 1) / warm
    p = (s - warm) / max(1, steps - warm)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def make_batch(rng, a, facts, train):
    nt = a.micro_bs // 2
    rows, ans = [], []
    for i in rng.integers(0, len(train.d) - a.seq - 1, nt):
        rows.append(np.asarray(train.d[i:i + a.seq + 1], dtype=np.int64))
    for b in range(a.micro_bs - nt):
        toks, an = facts.seq(rng, a.seq)
        rows.append(np.asarray(toks, dtype=np.int64))
        for t, c0, c1, rr, o in an:
            ans.append((nt + b, t, c0, c1, rr, o))
    arr = torch.from_numpy(np.stack(rows)).to(DEV)
    A = torch.tensor(ans, dtype=torch.long, device=DEV)
    return arr[:, :-1], arr[:, 1:], A


def step_loss(m, a, reader, x, y, A):
    h, aux, z = m.hidden(x)
    te = y >= a.vocab
    isl = m.isent(h).squeeze(-1).float()
    l_bin = F.binary_cross_entropy_with_logits(isl, te.float())
    l_text = F.cross_entropy(F.linear(h[~te], m.lm.emb.weight).float(),
                             y[~te])
    lp, ex = m.answer(h, A[:, 0], A[:, 1], A[:, 2:4], x, reader)
    l_ent = -lp.gather(1, A[:, 5:6]).mean()
    loss = l_text + l_bin + l_ent + a.aux_coef * aux + a.z_coef * z
    info = {}
    if ex:
        loss = loss + F.cross_entropy(ex["rel"], A[:, 4]) + F.cross_entropy(
            ex["ptr"], torch.zeros_like(A[:, 0]))
        info = {"gate": ex["gate"].mean().detach(),
                "rel": (ex["rel"].argmax(-1) == A[:, 4]).float().mean()}
    return loss, l_text.detach(), l_ent.detach(), info


@torch.no_grad()
def queries(m, reader, facts, qs, blank=False, bs=256):
    m.eval()
    V = facts.V
    preds, gates, rels = [], [], []
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        rows, at, sp = [], [], []
        for k, (s, rr) in enumerate(ch):
            tp = facts.tpl[rr]
            pre, mid, _ = tp[(i + k) % len(tp)]
            rows.append(pre + [V + s] + mid)
            at.append(len(rows[-1]) - 1)
            sp.append(len(pre))
        T = max(len(r) for r in rows)
        x = torch.zeros(len(rows), T, dtype=torch.long)
        for k, r in enumerate(rows):
            x[k, :len(r)] = torch.tensor(r)
        x = x.to(DEV)
        ab = torch.arange(len(rows), device=DEV)
        at_t = torch.tensor(at, device=DEV)
        cpos = torch.tensor([[p, -1] for p in sp], device=DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            h = m.hidden(x)[0]
            lp, ex = m.answer(h, ab, at_t, cpos, x, reader, blank)
        preds += lp.argmax(-1).tolist()
        if ex:
            gates += ex["gate"].tolist()
            rels += ex["rel"].argmax(-1).tolist()
    m.train()
    return preds, gates, rels


@torch.no_grad()
def text_val(m, val, a, iters=50):
    m.eval()
    rng = np.random.default_rng(1234)
    tot = 0.0
    for _ in range(iters):
        x, y = val.batch(rng, a.micro_bs)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            h = m.hidden(x)[0]
            lg = F.linear(h, m.lm.emb.weight)
        tot += F.cross_entropy(lg.float().view(-1, a.vocab),
                               y.view(-1)).item()
    m.train()
    return tot / iters


def preflight(a, base_sd, facts, train, val, reader, Afeat):
    m = MemLM(a, Afeat, True).to(DEV)
    m.lm.load_state_dict(base_sd)
    x, y, A = make_batch(np.random.default_rng(99), a, facts, train)
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        loss, _, _, _ = step_loss(m, a, reader, x, y, A)
    loss.backward()
    qs = facts.train_keys[:8] + facts.held_list[:8]
    queries(m, reader, facts, qs)
    queries(m, reader, facts, qs, blank=True)
    text_val(m, val, a, iters=1)
    log("preflight ok: loss %.2f, %d answer positions in one micro-batch"
        % (loss.item(), A.shape[0]))
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()


def train_model(name, memory, a, base_sd, facts, train, reader, Afeat):
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    ck = os.path.join(out, "ckpt.pt")
    torch.manual_seed(a.seed)
    m = MemLM(a, Afeat, memory).to(DEV)
    m.lm.load_state_dict(base_sd)
    new = sum(p.numel() for n_, p in m.named_parameters()
              if not n_.startswith("lm."))
    tps = a.micro_bs * a.accum * a.seq
    steps = a.tokens // tps
    decay = [p for p in m.parameters() if p.dim() >= 2]
    other = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=AMP)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    start = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEV)
        m.load_state_dict(st["model"])
        if st.get("done"):
            log("%s: already trained, loaded" % name)
            return m
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start = st["step"]
        log("%s: resuming at step %d" % (name, start))
    rng = np.random.default_rng(a.seed * 1000 + start)
    log("%s: %d steps of %d tokens (%.0fM tokens); new parameters %.2fM"
        % (name, steps, tps, steps * tps / 1e6, new / 1e6))
    sync()
    t_last, n_tok = time.time(), 0
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(a, step, steps)
        for _ in range(a.accum):
            x, y, A = make_batch(rng, a, facts, train)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss, lt, le, info = step_loss(m, a, reader, x, y, A)
            scaler.scale(loss / a.accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        n_tok += tps
        if (step + 1) % a.log_every == 0:
            sync()
            dt = time.time() - t_last
            t_last = time.time()
            rate = n_tok / dt
            n_tok = 0
            msg = ("%s step %d/%d text %.3f fact %.3f  %.0fk tok/s  ETA %.0f min"
                   % (name, step + 1, steps, lt.item(), le.item(), rate / 1e3,
                      (steps - step - 1) * tps / rate / 60))
            if info:
                msg += "  gate %.2f  relation acc %.2f" % (
                    info["gate"].item(), info["rel"].item())
            log(msg)
        last = step + 1 == steps
        if (step + 1) % a.ckpt_every == 0 or last:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": step + 1,
                        "done": last}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    del opt
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/moe_data")
    ap.add_argument("--kb", default="/content/mqa")
    ap.add_argument("--base", default="/content/drive/MyDrive/moe/runs/moe/ckpt.pt")
    ap.add_argument("--out", default="/content/drive/MyDrive/moe/memexp")
    ap.add_argument("--tokens", type=int, default=30_000_000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--ffn_dense", type=int, default=2048)
    ap.add_argument("--ffn_moe", type=int, default=1024)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--micro_bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux_coef", type=float, default=0.01)
    ap.add_argument("--z_coef", type=float, default=0.001)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--hold", type=float, default=0.1)
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--edits", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    a.vocab = tok.get_vocab_size()
    names, idx, rels, trip, bad = M.load_kb(a.kb)
    NB = len(rels)
    NREL = 2 * NB
    a.nrel = NREL
    S, Rr, O = [], [], []
    for s, r, o in trip:
        S += [s, o]
        Rr += [r, r + NB]
        O += [o, s]
    mem = M4.EditMem(len(names), NREL, a.D, 0)
    mem.build(S, Rr, O, a.L, len(names), NREL)
    reader = Reader(mem, NREL, a.L, a.cap)
    Afeat = torch.cat([mem.addr.real, mem.addr.imag], -1).to(
        torch.float16 if AMP else torch.float32).contiguous()
    facts = Facts(tok, rels, trip, NB, a.vocab, a.hold, a.seed)
    log("KB: %d entities, %d facts in memory (%d with inverses), %d banks; "
        "%d of %d keys (%d facts) held out of training text"
        % (len(names), len(trip), len(S), mem.nb, len(facts.held),
           facts.n_keys, facts.n_held_facts))
    train = MT.Data(os.path.join(a.data, "train.bin"), a.seq)
    val = MT.Data(os.path.join(a.data, "val.bin"), a.seq)
    base_sd = torch.load(a.base, map_location=DEV)["model"]
    preflight(a, base_sd, facts, train, val, reader, Afeat)
    models = {}
    for name, use in (("baseline", False), ("memory", True)):
        models[name] = train_model(name, use, a, base_sd, facts, train,
                                   reader, Afeat)

    rng = random.Random(a.seed + 7)
    seen = rng.sample(facts.train_keys, min(a.n_eval, len(facts.train_keys)))
    unseen = rng.sample(facts.held_list, min(a.n_eval, len(facts.held_list)))
    cnt = collections.defaultdict(collections.Counter)
    for s, r, o in facts.fwd.tolist():
        cnt[r][o] += 1
    prior = {r: c.most_common(1)[0][0] for r, c in cnt.items()}
    head = {i for i, n_ in enumerate(rels) if n_ in HEAD}

    gref = [facts.obj]

    def acc(preds, qs, sub=None):
        ok = [p in gref[0][q] for p, q in zip(preds, qs)
              if sub is None or q[1] in sub]
        return sum(ok) / max(len(ok), 1)

    R = {}
    for name, m in models.items():
        R[name, "text"] = text_val(m, val, a)
        for tag, qs in (("seen", seen), ("unseen", unseen)):
            p, g, rl = queries(m, reader, facts, qs)
            R[name, tag] = p
            if g:
                R[name, tag, "gate"] = sum(g) / len(g)
                R[name, tag, "rel"] = sum(int(x_ == q[1]) for x_, q in
                                          zip(rl, qs)) / len(qs)
                R[name, tag, "blank"] = queries(m, reader, facts, qs,
                                                blank=True)[0]
    log("evaluated")
    gold0 = {q: set(facts.obj[q]) for q in seen + unseen}

    def pick(qs):
        c = [q for q in qs if q[1] in head and len(facts.obj[q]) == 1]
        return c[:a.edits]
    ed = {"seen": pick(seen), "unseen": pick(unseen)}
    pool = {r: sorted({o for s, rr, o in trip if rr == r}) for r in head}
    erng = random.Random(a.seed + 11)
    edits = []
    for tag in ("seen", "unseen"):
        for s, r in ed[tag]:
            o = next(iter(facts.obj[(s, r)]))
            o2 = o
            while o2 == o:
                o2 = erng.choice(pool[r])
            edits.append((tag, s, r, o, o2))
    sync()
    t0 = time.time()
    for _, s, r, o, o2 in edits:
        mem.change(s, r, o, o2, NB)
    sync()
    t_edit = time.time() - t0
    reader.refresh()
    for _, s, r, o, o2 in edits:
        facts.obj[(s, r)] = {o2}
        facts.obj[(o, r + NB)].discard(s)
        facts.obj[(o2, r + NB)].add(s)
    edited = {(s, r) for _, s, r, _, _ in edits}
    un = [q for q in seen + unseen if q not in edited]
    E = {}
    for name, m in models.items():
        for tag in ("seen", "unseen"):
            p = queries(m, reader, facts, ed[tag])[0]
            new = [e[4] for e in edits if e[0] == tag]
            old = [e[3] for e in edits if e[0] == tag]
            n_ = max(len(p), 1)
            E[name, tag, "new"] = sum(x_ == y_ for x_, y_ in zip(p, new)) / n_
            E[name, tag, "old"] = sum(x_ == y_ for x_, y_ in zip(p, old)) / n_
        before = dict(zip(seen + unseen, R[name, "seen"] + R[name, "unseen"]))
        E[name, "un_before"] = acc([before[q] for q in un], un)
        E[name, "un_after"] = acc(queries(m, reader, facts, un)[0], un)
    log("edit test done")

    def f3(v):
        return "%.3f" % v if v is not None else "-"

    def row(label, b, mm):
        print("%-54s %10s %13s" % (label, f3(b), f3(mm)))

    gref[0] = gold0
    pri = {"seen": [prior.get(q[1], -1) for q in seen],
           "unseen": [prior.get(q[1], -1) for q in unseen]}
    print()
    print("=== MoE step 2: memory expert vs MoE alone (%dM tokens each, "
          "from the step-1 MoE) ===" % (a.tokens // 1_000_000))
    print("knowledge: %d MetaQA facts, all in the frozen memory; %d of %d "
          "(film, relation) keys (%d facts) never appear in training text"
          % (len(trip), len(facts.held), facts.n_keys, facts.n_held_facts))
    print("eval: %d keys seen in text, %d keys only in memory; Hits@1 = the "
          "top entity is a correct answer; DWAY = director, writer, actor, "
          "year" % (len(seen), len(unseen)))
    print()
    print("%-54s %10s %13s" % ("", "MoE alone", "MoE + memory"))
    row("FineWeb text loss (val; step-1 MoE was 4.216)",
        R["baseline", "text"], R["memory", "text"])
    for tag, lab in (("seen", "Facts seen in training text"),
                     ("unseen", "Facts ONLY in memory")):
        qs = seen if tag == "seen" else unseen
        row(lab + ", all relations", acc(R["baseline", tag], qs),
            acc(R["memory", tag], qs))
        row(lab + ", DWAY", acc(R["baseline", tag], qs, head),
            acc(R["memory", tag], qs, head))
        row("  most-common-answer guess, all / DWAY", acc(pri[tag], qs),
            acc(pri[tag], qs, head))
        row("  memory blanked, all relations", None,
            acc(R["memory", tag, "blank"], qs))
        print("  MoE + memory: gate (memory share) %.3f, controller "
              "relation accuracy %.3f" % (R["memory", tag, "gate"],
                                          R["memory", tag, "rel"]))
    print()
    print("EDITS: %d single-answer DWAY facts changed in memory in %.3f s "
          "(%.2f ms each); nothing retrained"
          % (len(edits), t_edit, 1000 * t_edit / max(len(edits), 1)))
    print("%-54s %10s %13s" % ("", "MoE alone", "MoE + memory"))
    for tag, lab in (("seen", "seen in text"), ("unseen", "only in memory")):
        row("Edited facts %s (%d): NEW answer" % (lab, len(ed[tag])),
            E["baseline", tag, "new"], E["memory", tag, "new"])
        row("Edited facts %s: OLD answer" % lab,
            E["baseline", tag, "old"], E["memory", tag, "old"])
    row("Unedited facts before edits", E["baseline", "un_before"],
        E["memory", "un_before"])
    row("Unedited facts after edits", E["baseline", "un_after"],
        E["memory", "un_after"])
    print()
    print("PER RELATION, facts only in memory (Hits@1)")
    print("%-10s %5s %7s %10s %13s" % ("relation", "n", "guess", "MoE alone",
                                       "MoE + memory"))
    for r in sorted({q[1] for q in unseen}):
        sub = {r}
        n_ = sum(1 for q in unseen if q[1] == r)
        print("%-10s %5d %7.3f %10.3f %13.3f"
              % (M.ABBR.get(rels[r], rels[r][:10]), n_,
                 acc(pri["unseen"], unseen, sub),
                 acc(R["baseline", "unseen"], unseen, sub),
                 acc(R["memory", "unseen"], unseen, sub)))
    res = {"|".join(map(str, k)): v for k, v in list(R.items()) + list(E.items())
           if not isinstance(v, list)}
    res["edit_seconds"] = t_edit
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
