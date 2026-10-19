"""
pmlm2.py - PhaseMem-LM Stage 2: natural phrasing, 2-token names,
mixed questions, unknown answers, and a facts-in-context (RAG)
baseline.
World: people / companies / cities.
  works_at P->C, lives_in P->T, located_in C->T, ceo C->P
  (5% of facts missing), friend P->P (0-3 friends).
Names: 2 syllable tokens (128 x 128 = 16,384 possible names).
Address of a name = S1[syl1] * S2[syl2]; memory is never trained.
Questions: memory (1-4 hops, nested "the X of the Y of NAME"),
  copy ("repeat the name NAME"), context ("note that A works at B .
  what is the employer of A ?" - text overrides memory).
Answer: an entity of the world, or "unknown".
Models (same Transformer): OFF (memorizes one world), RAG (facts
  pasted into the input), ON (banked phase memory + controller).
"""
import argparse
import copy
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NR = 5
DOM = [0, 0, 1, 0, 1]
RNG = [1, 2, 2, 0, 0]
FRIEND = 3
OUT = {0: [0, 1, 3], 1: [2, 4]}
CONT = {0: [0, 3], 1: [4]}
HMAX = 4
MAXLEN = 128
QNP = [["the employer of", "the company of", "the workplace of"],
       ["the home city of", "the hometown of",
        "the city of residence of"],
       ["the location of", "the base city of",
        "the headquarters city of"],
       ["a friend of", "one friend of", "some friend of"],
       ["the ceo of", "the chief of", "the leader of"]]
FACT = [["works at", "is employed by", "has a job at"],
        ["lives in", "resides in", "is at home in"],
        ["is located in", "is based in", "sits in"],
        ["is friends with", "knows", "is a pal of"],
        ["is run by", "is led by", "is headed by"]]
OPEN = ["what is", "who is", "tell me", "name", "do you know"]
COPY = ["repeat the name", "say back the name", "echo the name"]
NOTE = ["note that", "update :", "remember that"]
FILL = ["please", "hey", "quick question :"]
EXTRA = ["thanks", "?", ".", "facts", ":", "question"]
CONS = "b c d f g h k l m n p r s t v z".split()
VOW = "a e i o u y ae oo".split()
SYL = ["#" + c + v for c in CONS for v in VOW]
NSYL = len(SYL)


def _words():
    w = set()
    for lst in QNP + FACT:
        for p in lst:
            w.update(p.split())
    for p in OPEN + COPY + NOTE + FILL + EXTRA:
        w.update(p.split())
    return sorted(w)


VOCAB = ["<pad>", "<a>"] + _words() + SYL
TOK = {w: i for i, w in enumerate(VOCAB)}
SYL0 = len(VOCAB) - NSYL


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


class World:
    def __init__(self, n, rng, miss=0.05):
        npl, nco = n // 2, n // 4
        self.n = n
        self.groups = [list(range(0, npl)),
                       list(range(npl, npl + nco)),
                       list(range(npl + nco, n))]
        self.type = [0] * npl + [1] * nco + [2] * (n - npl - nco)
        codes = rng.sample(range(NSYL * NSYL), n)
        self.name = [(v // NSYL, v % NSYL) for v in codes]
        self.obj = {}
        for r in range(NR):
            dom, ran = self.groups[DOM[r]], self.groups[RNG[r]]
            for s in dom:
                if r == FRIEND:
                    k = rng.choice([0, 1, 1, 2, 2, 3])
                    fr = rng.sample(ran, k + 1)
                    self.obj[(s, r)] = [o for o in fr if o != s][:k]
                elif rng.random() >= miss:
                    self.obj[(s, r)] = [rng.choice(ran)]


def prepare(world, mem):
    nt = torch.tensor(world.name, device=DEV)
    world.name_t = nt
    world.addr = mem.S1[nt[:, 0]] * mem.S2[nt[:, 1]]


def follow(world, s, rels):
    cur = {s}
    for r in rels:
        nxt = set()
        for c in cur:
            nxt.update(world.obj.get((c, r), ()))
        cur = nxt
    return cur


def one_chain(world, s, rels):
    if not rels:
        return []
    for o in world.obj.get((s, rels[0]), ()):
        rest = one_chain(world, o, rels[1:])
        if rest is not None:
            return [(s, rels[0], o)] + rest
    return None


def path_facts(world, s, rels):
    out, cur = [], {s}
    for r in rels:
        nxt = set()
        for c in cur:
            for o in world.obj.get((c, r), ()):
                out.append((c, r, o))
                nxt.add(o)
        cur = nxt
    return out


def name_toks(world, e):
    a, b = world.name[e]
    return [SYL[a], SYL[b]]


def q_text(world, rng, s, rels):
    words = rng.choice(OPEN).split()
    for r in reversed(rels):
        words += rng.choice(QNP[r]).split()
    words += name_toks(world, s) + ["?"]
    if rng.random() < 0.3:
        words = rng.choice(FILL).split() + words
    if rng.random() < 0.2:
        words = words + ["thanks"]
    return words, follow(world, s, rels), (s, rels)


def q_memory(world, rng, h):
    t = rng.choice([0, 1])
    s = rng.choice(world.groups[t])
    rels = []
    for i in range(h):
        opts = OUT if i == h - 1 else CONT
        r = rng.choice(opts[t])
        rels.append(r)
        t = RNG[r]
    return q_text(world, rng, s, rels)


def q_copy(world, rng):
    e = rng.randrange(world.n)
    words = rng.choice(COPY).split() + name_toks(world, e) + ["?"]
    return words, {e}, None


def q_context(world, rng):
    t = rng.choice([0, 1])
    s = rng.choice(world.groups[t])
    r = rng.choice([x for x in OUT[t] if x != FRIEND])
    o = rng.choice(world.groups[RNG[r]])
    words = (rng.choice(NOTE).split() + name_toks(world, s)
             + rng.choice(FACT[r]).split() + name_toks(world, o)
             + ["."] + rng.choice(OPEN).split()
             + rng.choice(QNP[r]).split() + name_toks(world, s)
             + ["?"])
    return words, {o}, (s, [r])


def held(key):
    s, rels = key
    return len(rels) >= 2 and hash((s,) + tuple(rels)) % 10 == 0


def sample_items(world, rng, B, holdout):
    items = []
    while len(items) < B:
        u = rng.random()
        if u < 0.7:
            it = q_memory(world, rng, rng.randint(1, HMAX))
            if holdout and held(it[2]):
                continue
        elif u < 0.8:
            it = q_copy(world, rng)
        else:
            it = q_context(world, rng)
        items.append(it)
    return items


def rag_prefix(world, rng, key, nfacts=8):
    facts = []
    if key is not None:
        s, rels = key
        ch = one_chain(world, s, rels)
        facts = ch if ch is not None else path_facts(world, s, rels)
    facts = list(facts[:nfacts])
    while len(facts) < nfacts:
        r = rng.randrange(NR)
        s = rng.choice(world.groups[DOM[r]])
        objs = world.obj.get((s, r), ())
        if objs:
            facts.append((s, r, rng.choice(objs)))
    rng.shuffle(facts)
    words = ["facts", ":"]
    for s, r, o in facts:
        words += (name_toks(world, s) + rng.choice(FACT[r]).split()
                  + name_toks(world, o) + ["."])
    return words + ["question", ":"]


def encode(world, items, rag, rng):
    seqs = []
    for words, ans, key in items:
        pre = rag_prefix(world, rng, key) if rag else []
        seqs.append([TOK[w] for w in pre + words + ["<a>"]])
    T = max(len(s) for s in seqs)
    assert T <= MAXLEN
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    apos = torch.tensor([len(s) - 1 for s in seqs])
    tgt = torch.zeros(len(seqs), world.n + 1, dtype=torch.bool)
    for i, (_, ans, _) in enumerate(items):
        if ans:
            tgt[i, list(ans)] = True
        else:
            tgt[i, world.n] = True
    return x.to(DEV), apos.to(DEV), tgt.to(DEV)


class Memory:
    def __init__(self, D, L, seed):
        g = torch.Generator().manual_seed(seed)

        def ph(n):
            u = torch.rand(n, D, generator=g) * 2 - 1
            return torch.exp(1j * u * math.pi).to(DEV)
        self.S1, self.S2 = ph(NSYL), ph(NSYL)
        self.R = ph(NR)
        self.P1, self.P2 = ph(31), ph(31)
        self.pow2 = (2 ** torch.arange(31, device=DEV)).long()
        self.D, self.L = D, L

    def route(self, key, nb):
        out = []
        for P in (self.P1, self.P2):
            bits = ((key @ P.conj().T).real > 0).long()
            out.append((bits * self.pow2).sum(-1) % nb)
        return out

    @torch.no_grad()
    def write(self, world):
        trip = [(s, r, o) for (s, r), objs in world.obj.items()
                for o in objs]
        s = torch.tensor([t[0] for t in trip], device=DEV)
        r = torch.tensor([t[1] for t in trip], device=DEV)
        o = torch.tensor([t[2] for t in trip], device=DEV)
        nb = max(len(trip) // self.L, 1)
        key = world.addr[s] * self.R[r]
        b1, b2 = self.route(key, nb)
        load = [0] * nb
        fb = []
        for x1, x2 in zip(b1.tolist(), b2.tolist()):
            b = x2 if load[x2] < load[x1] else x1
            load[b] += 1
            fb.append(b)
        fb = torch.tensor(fb, device=DEV)
        fact = key * torch.roll(world.addr[o], 1, -1)
        re = torch.zeros(nb, self.D, device=DEV)
        im = torch.zeros(nb, self.D, device=DEV)
        re.index_add_(0, fb, fact.real.contiguous())
        im.index_add_(0, fb, fact.imag.contiguous())
        self.B = torch.complex(re, im)
        self.nb, self.n = nb, world.n
        self.Ac = world.addr.conj().T.contiguous()

    @torch.no_grad()
    def read(self, addr):
        key = addr[:, None, :] * self.R[None]
        k = key.reshape(-1, self.D)
        b1, b2 = self.route(k, self.nb)
        kc = k.conj()
        s1 = (torch.roll(self.B[b1] * kc, -1, -1) @ self.Ac).real
        s2 = (torch.roll(self.B[b2] * kc, -1, -1) @ self.Ac).real
        pick = (s2.amax(1) > s1.amax(1))[:, None]
        s = torch.where(pick, s2, s1) / self.D
        return s.view(-1, NR, self.n)


class Model(nn.Module):
    def __init__(self, a, use_mem):
        super().__init__()
        d = a.d
        self.use_mem, self.K, self.d, self.D = use_mem, a.K, d, a.D
        self.tok = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(MAXLEN, d)
        layer = nn.TransformerEncoderLayer(
            d, a.heads, 4 * d, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, a.layers,
                                        enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.u1 = nn.Embedding(NSYL, d)
        self.u2 = nn.Embedding(NSYL, d)
        self.unk = nn.Linear(d, 1)
        m = torch.triu(torch.full((MAXLEN, MAXLEN), float("-inf")), 1)
        self.register_buffer("causal", m)
        if use_mem:
            self.hop = nn.Parameter(torch.randn(HMAX, d) * 0.02)
            self.rq = nn.Linear(d, d)
            self.rk = nn.Linear(d, d)
            self.sq = nn.Linear(d, d)
            self.sk = nn.Linear(d, d)
            self.rel_head = nn.Linear(d, NR)
            self.prev_head = nn.Linear(d, 1)
            self.halt_head = nn.Linear(d, 1)
            self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                     nn.Linear(d, d))
            self.gate = nn.Linear(d, 1)
            self.unk_mem = nn.Linear(d + 1, 1)
            self.log_beta = nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(self, x, apos, world):
        T = x.shape[1]
        e = self.tok(x) + self.pos.weight[None, :T]
        h = self.norm(self.tf(e, mask=self.causal[:T, :T]))
        hl = h[torch.arange(len(x), device=DEV), apos]
        nt = world.name_t
        E = self.u1(nt[:, 0]) + self.u2(nt[:, 1])
        ent = self.q(hl) @ E.T / math.sqrt(self.d)
        lm = torch.cat([ent, self.unk(hl)], -1)
        if not self.use_mem:
            return F.log_softmax(lm, -1), {}
        p_mem = self.controller(h, x, hl, world)
        g = torch.sigmoid(self.gate(hl))
        p = g * p_mem + (1 - g) * F.softmax(lm, -1)
        return torch.log(p + 1e-9), {"gate": g.mean().item()}

    def controller(self, h, x, hl, world):
        mem = self.mem
        B, T, d = h.shape
        valid = x != 0
        is_syl = x >= SYL0
        nxt = torch.zeros_like(is_syl)
        nxt[:, :-1] = is_syl[:, 1:]
        start_ok = is_syl & nxt
        a_idx = (x - SYL0).clamp(min=0)
        b_idx = torch.zeros_like(a_idx)
        b_idx[:, :-1] = a_idx[:, 1:]
        sl = torch.einsum("bd,btd->bt", self.sq(hl), self.sk(h))
        sl = (sl / math.sqrt(d)).masked_fill(~start_ok, -1e9)
        tw, tpos = sl.softmax(-1).topk(2, -1)
        tw = tw / tw.sum(-1, keepdim=True)
        qaddr = (mem.S1[a_idx.gather(1, tpos)]
                 * mem.S2[b_idx.gather(1, tpos)])
        keys = self.rk(h)
        beta = self.log_beta.exp()
        A = world.addr
        state = hl
        prev_ids = prev_p = None
        hop_p, halts, conf = [], [], []
        for i in range(HMAX):
            q = self.rq(state + self.hop[i])
            att = torch.einsum("bd,btd->bt", q, keys) / math.sqrt(d)
            att = att.masked_fill(~valid, -1e9).softmax(-1)
            ctx = torch.einsum("bt,btd->bd", att, h)
            rel_w = self.rel_head(ctx).softmax(-1)
            if prev_ids is None:
                addr, sw = qaddr, tw
            else:
                pq = torch.sigmoid(self.prev_head(state))
                addr = torch.cat([qaddr, A[prev_ids]], 1)
                sw = torch.cat([(1 - pq) * tw, pq * prev_p], 1)
            C = addr.shape[1]
            S = mem.read(addr.reshape(-1, self.D))
            S = S.view(B, C, NR, -1)
            P = (beta * S).softmax(-1)
            p_i = torch.einsum("bc,br,bcrn->bn", sw, rel_w, P)
            best = S.amax(-1)
            conf.append(torch.einsum("bc,br,bcr->b", sw, rel_w, best))
            hop_p.append(p_i)
            halts.append(self.halt_head(state))
            tp, ti = p_i.topk(self.K, -1)
            prev_p = tp / tp.sum(-1, keepdim=True)
            prev_ids = ti
            state = state + self.upd(torch.cat([state, ctx], -1))
        alpha = torch.cat(halts, -1).softmax(-1)
        p_ent = torch.einsum("bh,bhn->bn", alpha,
                             torch.stack(hop_p, 1))
        c = (alpha * torch.stack(conf, -1)).sum(-1, keepdim=True)
        u = torch.sigmoid(self.unk_mem(torch.cat([hl, c], -1)))
        return torch.cat([(1 - u) * p_ent, u], -1)


def train(model, mem, a, kind, rng, world=None):
    rag, fixed = kind == "rag", kind == "off"
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    every = max(a.steps // 8, 1)
    for step in range(1, a.steps + 1):
        if not fixed and (step == 1 or step % a.world_every == 0):
            world = World(a.n_train, rng)
            prepare(world, mem)
            if kind == "on":
                mem.write(world)
        items = sample_items(world, rng, a.bs, fixed)
        x, apos, tgt = encode(world, items, rag, rng)
        logp, info = model(x, apos, world)
        loss = -logp.masked_fill(~tgt, -1e9).logsumexp(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % every == 0:
            pred = logp.argmax(-1)
            acc = tgt.gather(1, pred[:, None]).float().mean().item()
            log("  %s step %d loss=%.3f acc=%.3f gate=%.2f"
                % (kind, step, loss.item(), acc,
                   info.get("gate", float("nan"))))


@torch.no_grad()
def accuracy(model, world, items, rag, seed, bs=64):
    if not items:
        return float("nan")
    rng = random.Random(seed)
    model.eval()
    ok = 0
    for i in range(0, len(items), bs):
        x, apos, tgt = encode(world, items[i:i + bs], rag, rng)
        logp, _ = model(x, apos, world)
        pred = logp.argmax(-1)
        ok += tgt.gather(1, pred[:, None]).sum().item()
    model.train()
    return ok / len(items)


def mem_items(world, seed, h, nq, heldout):
    rng = random.Random(seed)
    out = []
    while len(out) < nq:
        it = q_memory(world, rng, h)
        if heldout and h >= 2 and not held(it[2]):
            continue
        out.append(it)
    return out


def edit(world, frac, rng):
    w2 = copy.copy(world)
    w2.obj = dict(world.obj)
    comps = world.groups[1]
    chosen = rng.sample(world.groups[0],
                        int(frac * len(world.groups[0])))
    for s in chosen:
        old = world.obj.get((s, 0), [None])[0]
        new = old
        while new == old:
            new = rng.choice(comps)
        w2.obj[(s, 0)] = [new]
    return w2, chosen


def fmt(v):
    return "  ".join("%5.3f" % x for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_train", type=int, default=1024)
    ap.add_argument("--world_every", type=int, default=25)
    ap.add_argument("--big", default="4096,16384")
    ap.add_argument("--models", default="on,rag,off")
    ap.add_argument("--nq", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    mem = Memory(a.D, a.L, a.seed)
    W = World(a.n_train, random.Random(1000))
    prepare(W, mem)
    kinds = a.models.split(",")
    models = {}
    for kind in kinds:
        m = Model(a, kind == "on").to(DEV)
        m.mem = mem
        npar = sum(p.numel() for p in m.parameters())
        log("%s: %d params, training" % (kind.upper(), npar))
        train(m, mem, a, kind, rng, world=W)
        models[kind] = m
        m.npar = npar

    def ev(kind, world, items, seed):
        return accuracy(models[kind], world, items, kind == "rag",
                        seed)

    mem.write(W)
    same = [mem_items(W, 100 + h, h, a.nq, True)
            for h in range(1, HMAX + 1)]
    unk = [it for hs in same for it in hs if not it[1]]
    r1 = random.Random(200)
    cpy = [q_copy(W, r1) for _ in range(a.nq)]
    r2 = random.Random(201)
    ctx = [q_context(W, r2) for _ in range(a.nq)]
    W2 = World(a.n_train, random.Random(2000))
    prepare(W2, mem)
    new = [mem_items(W2, 300 + h, h, a.nq, False)
           for h in range(1, HMAX + 1)]
    W3, chosen = edit(W, 0.2, random.Random(9))
    r3 = random.Random(400)
    e1 = [q_text(W3, r3, s, [0]) for s in chosen]
    e2 = [q_text(W3, r3, s, [0, 2]) for s in chosen]

    out = {}
    for k in kinds:
        mem.write(W)
        row = [ev(k, W, it, 1) for it in same]
        row += [ev(k, W, unk, 2), ev(k, W, cpy, 3), ev(k, W, ctx, 4)]
        mem.write(W2)
        nrow = [ev(k, W2, it, 5) for it in new]
        mem.write(W3)
        erow = [ev(k, W3, e1, 6), ev(k, W3, e2, 7)]
        out[k] = (row, nrow, erow)
        log("  evaluated %s" % k)

    big = []
    for n in [int(v) for v in a.big.split(",")]:
        Wb = World(n, random.Random(3000 + n))
        prepare(Wb, mem)
        mem.write(Wb)
        items = [mem_items(Wb, 500 + h, h, a.nq, False)
                 for h in range(1, HMAX + 1)]
        for k in kinds:
            if k == "off":
                continue
            big.append((n, k, [ev(k, Wb, it, 8) for it in items]))
        log("  evaluated %d-entity world" % n)

    print()
    print("=== PhaseMem-LM Stage 2 ===")
    for k in kinds:
        print("%-3s %d params" % (k.upper(), models[k].npar))
    print("vocab %d tokens; names are 2 syllable tokens" % len(VOCAB))
    print()
    print("SAME WORLD (%d entities; OFF trained on it; 2-4 hops are"
          " held-out compositions)" % a.n_train)
    print("model  1-hop  2-hop  3-hop  4-hop  unknwn   copy  contxt")
    for k in kinds:
        print("%-5s  %s" % (k.upper(), fmt(out[k][0])))
    print("(unknown: %d questions whose answer is missing)" % len(unk))
    print()
    print("NEW WORLD (all facts and names re-randomized,"
          " no retraining)")
    print("model  1-hop  2-hop  3-hop  4-hop")
    for k in kinds:
        print("%-5s  %s" % (k.upper(), fmt(out[k][1])))
    print()
    print("EDITS (20%% of people change employer, no retraining)")
    print("model  works_at  works_at->located_in")
    for k in kinds:
        e = out[k][2]
        print("%-5s  %8.3f  %8.3f" % (k.upper(), e[0], e[1]))
    print()
    print("BIGGER WORLDS (never trained at this size)")
    print(" size  model  1-hop  2-hop  3-hop  4-hop")
    for n, k, r in big:
        print("%5d  %-5s  %s" % (n, k.upper(), fmt(r)))
    log("done")


if __name__ == "__main__":
    main()
