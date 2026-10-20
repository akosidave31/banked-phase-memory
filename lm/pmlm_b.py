"""
pmlm_b.py - Track B v1: frozen SmolLM2-135M + banked phase memory.

The pretrained LM is frozen. A small trained parser (2 Transformer
layers on top of the LM's hidden states) reads a question and outputs
the subject name (start/end tokens), the number of hops, and the
relation at each hop.
Memory (never trained): fact = addr(subj) * R[rel] * shift(addr(obj)),
in banks of ~L facts routed by sign bits of the key (two-choice).
A name's address is computed from its words (hash -> phasor, bound by
position), so any name, seen or not, has an address.
Answering: follow the predicted relations with a top-K beam; answer
"unknown" if the weakest link scores below tau.
Parser training uses TRAINING wording only and names from new random
worlds. Tests use UNSEEN wording (different words), new worlds, edits,
bigger worlds. Baselines: the same LM prompted with no facts, with
facts + distractors, and with only the needed facts (oracle).
"""
import argparse
import copy
import hashlib
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NR = 5
DOM = [0, 0, 1, 0, 1]
RNG = [1, 2, 2, 0, 0]
FRIEND = 3
OUT = {0: [0, 1, 3], 1: [2, 4]}
CONT = {0: [0, 3], 1: [4]}
HMAX = 4
TR_NP = [["the employer of", "the company of"],
         ["the home city of", "the city of residence of"],
         ["the location of", "the headquarters city of"],
         ["a friend of", "one of the friends of"],
         ["the CEO of", "the boss of"]]
TE_NP = [["the firm that employs", "the workplace of"],
         ["the hometown of", "the town that is home to"],
         ["the base of", "the city hosting"],
         ["a pal of", "a buddy of"],
         ["the chief executive of", "the leader of"]]
TR_OPEN = ["What is", "Who is", "Tell me", "Name"]
TE_OPEN = ["Do you know", "Could you tell me"]
FACT_TXT = ["{} works at {}.", "{} lives in {}.",
            "{} is located in {}.", "{} is friends with {}.",
            "{} is run by {}."]
CONS = list("bcdfgklmnprstvz")
VOWS = list("aeiou")
CO_SUF = ["Corp", "Labs", "Systems", "Foods", "Motors", "Bank",
          "Media", "Works"]
CITY_PRE = ["Port", "San", "New", "Fort", "Lake", "Mount"]
FEW = ("Answer the question using only the facts. Reply with just "
       "the name, or unknown.\n\n"
       "Facts: Lino Mar works at Kepa Labs. Kepa Labs is located in "
       "Port Vel.\n"
       "Question: What is the location of the employer of Lino Mar?\n"
       "Answer: Port Vel\n\n"
       "Facts: Tia Ros lives in San Sora.\n"
       "Question: What is the employer of Tia Ros?\n"
       "Answer: unknown\n\n")


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def word(rng):
    k = rng.choice([2, 2, 3])
    w = "".join(rng.choice(CONS) + rng.choice(VOWS) for _ in range(k))
    return w.capitalize()


def make_name(rng, t):
    if t == 0:
        return word(rng) + " " + word(rng)
    if t == 1:
        return word(rng) + " " + rng.choice(CO_SUF)
    return rng.choice(CITY_PRE) + " " + word(rng)


class World:
    def __init__(self, n, rng, miss=0.05):
        npl, nco = n // 2, n // 4
        self.n = n
        self.groups = [list(range(0, npl)),
                       list(range(npl, npl + nco)),
                       list(range(npl + nco, n))]
        self.type = [0] * npl + [1] * nco + [2] * (n - npl - nco)
        seen, self.name = set(), []
        for t in self.type:
            s = make_name(rng, t)
            while s in seen:
                s = make_name(rng, t)
            seen.add(s)
            self.name.append(s)
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


class Addr:
    def __init__(self, D):
        self.D, self.cache = D, {}

    def word(self, w):
        v = self.cache.get(w)
        if v is None:
            seed = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            u = torch.rand(self.D, generator=g) * 2 - 1
            v = torch.exp(1j * u * math.pi)
            self.cache[w] = v
        return v

    def name(self, s):
        v = torch.ones(self.D, dtype=torch.complex64)
        for i, w in enumerate(s.split()):
            v = v * torch.roll(self.word(w), i)
        return v


def prepare(world, addr):
    world.addr = torch.stack([addr.name(s) for s in world.name]).to(DEV)


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


def make_q(world, rng, s, rels, test):
    nps = TE_NP if test else TR_NP
    parts = [rng.choice(TE_OPEN if test else TR_OPEN)]
    parts += [rng.choice(nps[r]) for r in reversed(rels)]
    prefix = " ".join(parts) + " "
    name = world.name[s]
    return dict(text=prefix + name + "?",
                span=(len(prefix), len(prefix) + len(name)),
                s=s, rels=rels, gold=follow(world, s, rels))


def q_memory(world, rng, h, test):
    t = rng.choice([0, 1])
    s = rng.choice(world.groups[t])
    rels = []
    for i in range(h):
        opts = OUT if i == h - 1 else CONT
        r = rng.choice(opts[t])
        rels.append(r)
        t = RNG[r]
    return make_q(world, rng, s, rels, test)


def qset(world, seed, h, nq, test):
    rng = random.Random(seed)
    return [q_memory(world, rng, h, test) for _ in range(nq)]


class Memory:
    def __init__(self, D, L, seed):
        g = torch.Generator().manual_seed(seed)

        def ph(n):
            u = torch.rand(n, D, generator=g) * 2 - 1
            return torch.exp(1j * u * math.pi).to(DEV)
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
        self.nb = nb
        self.Ac = world.addr.conj().T.contiguous()

    @torch.no_grad()
    def read(self, addr, rel):
        key = addr * self.R[rel]
        b1, b2 = self.route(key, self.nb)
        kc = key.conj()
        s1 = (torch.roll(self.B[b1] * kc, -1, -1) @ self.Ac).real
        s2 = (torch.roll(self.B[b2] * kc, -1, -1) @ self.Ac).real
        pick = (s2.amax(1) > s1.amax(1))[:, None]
        return torch.where(pick, s2, s1) / self.D


@torch.no_grad()
def beam_answer(mem, world, sa, rels, hops, K, tau):
    B = sa.shape[0]
    ca = sa[:, None, :]
    conf = torch.ones(B, 1, device=DEV)
    weak = torch.ones(B, 1, device=DEV)
    best = torch.full((B,), -1, dtype=torch.long, device=DEV)
    bweak = torch.zeros(B, device=DEV)
    for i in range(HMAX):
        C = ca.shape[1]
        r = rels[:, i][:, None].expand(B, C).reshape(-1)
        S = mem.read(ca.reshape(B * C, -1), r).view(B, C, -1)
        n = S.shape[-1]
        tot = conf[..., None] * S.clamp(min=0)
        v, idx = tot.view(B, -1).topk(K, -1)
        ci, ei = idx // n, idx % n
        sv = S.view(B, -1).gather(1, idx)
        weak = torch.minimum(weak.gather(1, ci), sv)
        conf = v
        ca = world.addr[ei]
        done = hops == i + 1
        best[done] = ei[done, 0]
        bweak[done] = weak[done, 0]
    return best, bweak < tau


class Parser(nn.Module):
    def __init__(self, hid, d=256, layers=2, maxlen=64):
        super().__init__()
        self.proj = nn.Linear(hid, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(
            d, 4, 4 * d, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers,
                                         enable_nested_tensor=False)
        self.start = nn.Linear(d, 1)
        self.end = nn.Linear(d, 1)
        self.slots = nn.Parameter(torch.randn(HMAX, d) * 0.02)
        self.att = nn.MultiheadAttention(d, 4, batch_first=True)
        self.rel = nn.Linear(d, NR)
        self.hops = nn.Linear(d, HMAX)

    def forward(self, f, mask):
        B, T, _ = f.shape
        h = self.proj(f) + self.pos.weight[None, :T]
        h = self.enc(h, src_key_padding_mask=~mask)
        st = self.start(h).squeeze(-1).masked_fill(~mask, -1e4)
        en = self.end(h).squeeze(-1).masked_fill(~mask, -1e4)
        q = self.slots[None].expand(B, -1, -1)
        o, _ = self.att(q, h, h, key_padding_mask=~mask)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1,
                                                          keepdim=True)
        return st, en, self.rel(o), self.hops(pooled)


@torch.no_grad()
def lm_feats(lm, enc):
    out = lm.model(input_ids=enc["input_ids"].to(DEV),
                   attention_mask=enc["attention_mask"].to(DEV))
    return out.last_hidden_state.float()


def tokenize(tok, qs):
    tok.padding_side = "right"
    return tok([q["text"] for q in qs], return_offsets_mapping=True,
               padding=True, return_tensors="pt")


def gold_spans(enc, qs):
    st, en = [], []
    for i, q in enumerate(qs):
        a, b = q["span"]
        o = enc["offset_mapping"][i].tolist()
        hit = [j for j, (x, y) in enumerate(o) if y > a and x < b]
        st.append(hit[0])
        en.append(hit[-1])
    return torch.tensor(st, device=DEV), torch.tensor(en, device=DEV)


def train_parser(parser, lm, tok, a, rng):
    opt = torch.optim.AdamW(parser.parameters(), lr=a.lr,
                            weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    world = None
    every = max(a.steps // 8, 1)
    for step in range(1, a.steps + 1):
        if step == 1 or step % 50 == 0:
            world = World(a.n_train, rng)
        qs = [q_memory(world, rng, rng.randint(1, HMAX), False)
              for _ in range(a.bs)]
        enc = tokenize(tok, qs)
        mask = enc["attention_mask"].bool().to(DEV)
        st_t, en_t = gold_spans(enc, qs)
        rel_t = torch.full((len(qs), HMAX), -100, dtype=torch.long)
        for i, q in enumerate(qs):
            rel_t[i, :len(q["rels"])] = torch.tensor(q["rels"])
        rel_t = rel_t.to(DEV)
        hop_t = torch.tensor([len(q["rels"]) - 1 for q in qs],
                             device=DEV)
        st, en, rl, hp = parser(lm_feats(lm, enc), mask)
        loss = (F.cross_entropy(st, st_t) + F.cross_entropy(en, en_t)
                + F.cross_entropy(rl.reshape(-1, NR), rel_t.reshape(-1),
                                  ignore_index=-100)
                + F.cross_entropy(hp, hop_t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parser.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % every == 0:
            ok = ((st.argmax(-1) == st_t) & (en.argmax(-1) == en_t))
            log("  step %d loss=%.3f span=%.3f hops=%.3f"
                % (step, loss.item(), ok.float().mean().item(),
                   (hp.argmax(-1) == hop_t).float().mean().item()))


@torch.no_grad()
def parse(parser, lm, tok, qs):
    parser.eval()
    enc = tokenize(tok, qs)
    mask = enc["attention_mask"].bool().to(DEV)
    st, en, rl, hp = parser(lm_feats(lm, enc), mask)
    T = st.shape[1]
    i = torch.arange(T, device=DEV)
    valid = (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] <= 8)
    sc = st[:, :, None] + en[:, None, :]
    sc = sc.masked_fill(~valid[None], -1e9)
    flat = sc.reshape(len(qs), -1).argmax(-1)
    s_i, e_i = (flat // T).tolist(), (flat % T).tolist()
    offs = enc["offset_mapping"]
    names = []
    for b, q in enumerate(qs):
        x0 = int(offs[b, s_i[b], 0])
        x1 = int(offs[b, e_i[b], 1])
        names.append(q["text"][x0:x1].strip(" ?.,"))
    parser.train()
    return names, rl.argmax(-1), hp.argmax(-1) + 1


def eval_on(parser, lm, tok, mem, addr, world, qs, a, bs=64):
    if not qs:
        return float("nan")
    ok = 0
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        names, rels, hops = parse(parser, lm, tok, ch)
        sa = torch.stack([addr.name(nm) for nm in names]).to(DEV)
        best, unk = beam_answer(mem, world, sa, rels, hops, a.K, a.tau)
        for q, b, u in zip(ch, best.tolist(), unk.tolist()):
            if not q["gold"]:
                ok += int(u)
            else:
                ok += int((not u) and b in q["gold"])
    return ok / len(qs)


def parser_diag(parser, lm, tok, world, qs, bs=64):
    n_ok = h_ok = 0
    r_ok, r_n = [0] * HMAX, [0] * HMAX
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        names, rels, hops = parse(parser, lm, tok, ch)
        for q, nm, rr, hh in zip(ch, names, rels.tolist(),
                                 hops.tolist()):
            n_ok += int(nm == world.name[q["s"]])
            h_ok += int(hh == len(q["rels"]))
            for k, r in enumerate(q["rels"]):
                r_n[k] += 1
                r_ok[k] += int(rr[k] == r)
    return (n_ok / len(qs), h_ok / len(qs),
            [o / max(m, 1) for o, m in zip(r_ok, r_n)])


def rag_text(world, rng, q, mode, nfacts=8):
    if mode == "none":
        return "(none)"
    ch = one_chain(world, q["s"], q["rels"])
    facts = list(ch if ch is not None else
                 path_facts(world, q["s"], q["rels"]))[:nfacts]
    while mode == "distract" and len(facts) < nfacts:
        r = rng.randrange(NR)
        s = rng.choice(world.groups[DOM[r]])
        objs = world.obj.get((s, r), ())
        if objs:
            facts.append((s, r, rng.choice(objs)))
    rng.shuffle(facts)
    return " ".join(FACT_TXT[r].format(world.name[s], world.name[o])
                    for s, r, o in facts)


@torch.no_grad()
def eval_prompt(lm, tok, world, qs, mode, seed, bs=16):
    rng = random.Random(seed)
    ok = 0
    tok.padding_side = "left"
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        prompts = [FEW + "Facts: %s\nQuestion: %s\nAnswer:"
                   % (rag_text(world, rng, q, mode), q["text"])
                   for q in ch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
        out = lm.generate(**enc, max_new_tokens=10, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        for t, q in zip(texts, ch):
            pred = t.strip().split("\n")[0].lower()
            gold = [world.name[g].lower() for g in q["gold"]]
            if not gold:
                ok += int("unknown" in pred)
            else:
                hit = any(g in pred for g in gold)
                ok += int(hit and "unknown" not in pred)
    return ok / len(qs)


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
    ap.add_argument("--lm", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_train", type=int, default=1024)
    ap.add_argument("--big", default="4096,16384")
    ap.add_argument("--nq", type=int, default=300)
    ap.add_argument("--nq_base", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.lm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.lm, torch_dtype=dtype)
    lm = lm.to(DEV).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    n_lm = sum(p.numel() for p in lm.parameters())
    parser = Parser(lm.config.hidden_size).to(DEV)
    n_p = sum(p.numel() for p in parser.parameters())
    log("device=%s LM %s (%d frozen params), parser %d params"
        % (DEV, a.lm, n_lm, n_p))
    train_parser(parser, lm, tok, a, rng)

    addr = Addr(a.D)
    mem = Memory(a.D, a.L, a.seed)
    W = World(a.n_train, random.Random(2000))
    prepare(W, addr)
    mem.write(W)
    tr = [qset(W, 100 + h, h, a.nq, False) for h in range(1, 5)]
    te = [qset(W, 200 + h, h, a.nq, True) for h in range(1, 5)]
    unk = [q for hs in te for q in hs if not q["gold"]]
    diag = parser_diag(parser, lm, tok, W,
                       [q for hs in te for q in hs])
    on_tr = [eval_on(parser, lm, tok, mem, addr, W, q, a) for q in tr]
    on_te = [eval_on(parser, lm, tok, mem, addr, W, q, a) for q in te]
    on_unk = eval_on(parser, lm, tok, mem, addr, W, unk, a)
    log("evaluated ON on a new world")

    W3, chosen = edit(W, 0.2, random.Random(9))
    mem.write(W3)
    r3 = random.Random(300)
    e1 = [make_q(W3, r3, s, [0], True) for s in chosen]
    e2 = [make_q(W3, r3, s, [0, 2], True) for s in chosen]
    ed = [eval_on(parser, lm, tok, mem, addr, W3, e1, a),
          eval_on(parser, lm, tok, mem, addr, W3, e2, a)]

    big = []
    for n in [int(v) for v in a.big.split(",")]:
        Wb = World(n, random.Random(3000 + n))
        prepare(Wb, addr)
        mem.write(Wb)
        qs = [qset(Wb, 400 + h, h, a.nq, True) for h in range(1, 5)]
        big.append((n, [eval_on(parser, lm, tok, mem, addr, Wb, q, a)
                        for q in qs], mem.nb * a.D))
        log("evaluated %d-entity world" % n)

    base_qs = [q[:a.nq_base] for q in te]
    base = {}
    for mode in ["none", "distract", "oracle"]:
        base[mode] = [eval_prompt(lm, tok, W, q, mode, 7)
                      for q in base_qs]
        log("evaluated prompting baseline: %s" % mode)

    print()
    print("=== Track B v1: SmolLM2-135M (frozen) + banked phase memory ===")
    print("LM %d frozen params; trained parser %d params" % (n_lm, n_p))
    print()
    print("PARSER on UNSEEN wording, new names:")
    print("  name exact %.3f   hop count %.3f   relation hop 1-4: %s"
          % (diag[0], diag[1], fmt(diag[2])))
    print()
    print("ON (memory), new %d-entity world" % a.n_train)
    print("wording          1-hop  2-hop  3-hop  4-hop")
    print("training words   %s" % fmt(on_tr))
    print("UNSEEN words     %s" % fmt(on_te))
    print("unknown answers (unseen words): %.3f  (%d questions)"
          % (on_unk, len(unk)))
    print()
    print("EDITS (20%% change employer, unseen words, no retraining)")
    print("  works_at %.3f   works_at->located_in %.3f" % (ed[0], ed[1]))
    print()
    print("BIGGER WORLDS (unseen words)")
    print(" size  1-hop  2-hop  3-hop  4-hop  memory numbers")
    for n, r, size in big:
        print("%5d  %s  %d" % (n, fmt(r), size))
    print()
    print("BASELINES: same LM, prompted (no training), unseen words, "
          "%d questions per hop" % a.nq_base)
    print("system                     1-hop  2-hop  3-hop  4-hop")
    print("LM alone (no facts)        %s" % fmt(base["none"]))
    print("LM + facts + distractors   %s" % fmt(base["distract"]))
    print("LM + only needed facts     %s" % fmt(base["oracle"]))
    print("ON (memory)                %s" % fmt(on_te))
    log("done")


if __name__ == "__main__":
    main()
