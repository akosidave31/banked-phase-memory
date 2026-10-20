"""
moe_train.py - small MoE language model vs a dense model with the same
active compute (step 1 of the memory-MoE plan).
  dense: d=512, 8 layers, 8 heads, one GELU feed-forward of width 2048
  moe  : same, but 8 experts of width 1024 each, top-2 routing per token
         (same active width 2048), Switch-style load-balancing loss and
         router z-loss
Both: RoPE attention, RMSNorm, tied embeddings, fp16 autocast, same data
order and token budget. Checkpoints in --out; reruns resume, and a model
with a finished result.json is skipped.
"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

T0 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.w).to(x.dtype)


def rope(x, cos, sin):
    T = x.shape[-2]
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.h = h

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(
            2, 0, 3, 1, 4)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MoE(nn.Module):
    def __init__(self, d, f, E, k):
        super().__init__()
        self.router = nn.Linear(d, E, bias=False)
        self.experts = nn.ModuleList([MLP(d, f) for _ in range(E)])
        self.E, self.k = E, k
        self.last_frac = torch.zeros(E)

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        logits = self.router(xf).float()
        probs = logits.softmax(-1)
        w, idx = probs.topk(self.k, -1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros(xf.shape, device=x.device, dtype=torch.float32)
        for e in range(self.E):
            rows, slot = (idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            y = self.experts[e](xf[rows])
            out.index_add_(0, rows, y.float() * w[rows, slot, None])
        frac = F.one_hot(idx, self.E).float().sum(1).mean(0) / self.k
        aux = self.E * (frac * probs.mean(0)).sum()
        z = torch.logsumexp(logits, -1).pow(2).mean()
        self.last_frac = frac.detach()
        return out.view(B, T, C).to(x.dtype), aux, z


class Block(nn.Module):
    def __init__(self, d, h, ffn, moe, E, k):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attn(d, h)
        self.moe = moe
        self.ff = MoE(d, ffn, E, k) if moe else MLP(d, ffn)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        if self.moe:
            y, aux, z = self.ff(self.n2(x))
            return x + y, aux, z
        return x + self.ff(self.n2(x)), 0.0, 0.0


class LM(nn.Module):
    def __init__(self, a, moe):
        super().__init__()
        self.emb = nn.Embedding(a.vocab, a.d)
        ffn = a.ffn_moe if moe else a.ffn_dense
        self.blocks = nn.ModuleList([Block(a.d, a.heads, ffn, moe,
                                           a.experts, a.topk)
                                     for _ in range(a.layers)])
        self.nf = RMSNorm(a.d)
        hd = a.d // a.heads
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        f = torch.outer(torch.arange(a.seq).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        for n, p in self.named_parameters():
            if n.endswith("o.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * a.layers))

    def forward(self, idx, tgt=None):
        x = self.emb(idx)
        aux = z = 0.0
        for b in self.blocks:
            x, a_, z_ = b(x, self.cos, self.sin)
            aux = aux + a_
            z = z + z_
        logits = F.linear(self.nf(x), self.emb.weight)
        loss = None
        if tgt is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                   tgt.view(-1))
        return logits, loss, aux, z


class Data:
    def __init__(self, path, seq):
        self.d = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq = seq

    def batch(self, rng, bs):
        ix = rng.integers(0, len(self.d) - self.seq - 1, bs)
        x = np.stack([self.d[i:i + self.seq] for i in ix]).astype(np.int64)
        y = np.stack([self.d[i + 1:i + 1 + self.seq]
                      for i in ix]).astype(np.int64)
        return (torch.from_numpy(x).to(DEV, non_blocking=True),
                torch.from_numpy(y).to(DEV, non_blocking=True))


def count(m, moe, a):
    total = sum(p.numel() for p in m.parameters())
    if not moe:
        return total, total
    exp = sum(p.numel() for b in m.blocks for p in b.ff.experts.parameters())
    return total, total - int(exp * (a.experts - a.topk) / a.experts)


@torch.no_grad()
def evaluate(m, val, a):
    m.eval()
    rng = np.random.default_rng(1234)
    tot = 0.0
    for _ in range(a.eval_iters):
        x, y = val.batch(rng, a.micro_bs)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            tot += m(x, y)[1].item()
    m.train()
    return tot / a.eval_iters


@torch.no_grad()
def sample(m, tok, a, prompt, n=60):
    m.eval()
    x = torch.tensor([tok.encode(prompt).ids], device=DEV)
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            logits = m(x[:, -a.seq:])[0]
        p = (logits[0, -1].float() / 0.8).softmax(-1)
        v, i = p.topk(40)
        nxt = i[torch.multinomial(v / v.sum(), 1)]
        x = torch.cat([x, nxt[None]], 1)
    m.train()
    return tok.decode(x[0].tolist())


def run(name, a, train, val, tok):
    moe = name == "moe"
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    resf = os.path.join(out, "result.json")
    if os.path.exists(resf) and not a.force:
        log("%s: already finished, loading result" % name)
        return json.load(open(resf))
    torch.manual_seed(a.seed)
    m = LM(a, moe).to(DEV)
    total, active = count(m, moe, a)
    log("%s: %.1fM parameters, %.1fM active per token"
        % (name, total / 1e6, active / 1e6))
    decay = [p for p in m.parameters() if p.dim() >= 2]
    other = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=AMP)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    tps = a.micro_bs * a.accum * a.seq
    steps = a.tokens // tps
    warm = max(1, int(0.02 * steps))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    ck = os.path.join(out, "ckpt.pt")
    start, hist = 0, []
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEV)
        m.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start, hist = st["step"], st["hist"]
        log("%s: resuming at step %d" % (name, start))
    rng = np.random.default_rng(a.seed)
    for _ in range(start * a.accum):
        rng.integers(0, 2, a.micro_bs)
    log("%s: %d steps of %d tokens (%.0fM tokens)"
        % (name, steps, tps, steps * tps / 1e6))
    sync()
    t_last, t_train, n_tok = time.time(), 0.0, 0
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(a.accum):
            x, y = train.batch(rng, a.micro_bs)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                _, loss, aux, z = m(x, y)
                tot = loss + a.aux_coef * aux + a.z_coef * z
            scaler.scale(tot / a.accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        n_tok += tps
        if (step + 1) % a.log_every == 0:
            sync()
            dt = time.time() - t_last
            t_train += dt
            t_last = time.time()
            rate = n_tok / dt
            n_tok = 0
            eta = (steps - step - 1) * tps / rate / 60
            msg = ("%s step %d/%d loss %.3f  %.0fk tok/s  ETA %.0f min"
                   % (name, step + 1, steps, loss.item(), rate / 1e3, eta))
            if moe:
                fr = torch.stack([b.ff.last_frac for b in m.blocks])
                msg += "  aux %.3f  expert use min %.3f max %.3f" % (
                    float(aux) / a.layers, fr.min().item(), fr.max().item())
            log(msg)
        last = step + 1 == steps
        if (step + 1) % a.eval_every == 0 or last:
            vl = evaluate(m, val, a)
            hist.append((step + 1, vl))
            log("%s step %d: val loss %.4f (perplexity %.1f)"
                % (name, step + 1, vl, math.exp(vl)))
        if (step + 1) % a.ckpt_every == 0 or last:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": step + 1,
                        "hist": hist}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    text = sample(m, tok, a, "The history of")
    res = {"name": name, "total_M": total / 1e6, "active_M": active / 1e6,
           "tokens_M": steps * tps / 1e6, "val_loss": hist[-1][1],
           "ppl": math.exp(hist[-1][1]), "hist": hist,
           "tok_s": steps * tps / max(t_train, 1e-9) if start == 0 else None,
           "sample": text}
    json.dump(res, open(resf, "w"), indent=1)
    del m, opt
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/moe_data")
    ap.add_argument("--out", default="/content/drive/MyDrive/moe/runs")
    ap.add_argument("--models", default="dense,moe")
    ap.add_argument("--tokens", type=int, default=50_000_000)
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
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--aux_coef", type=float, default=0.01)
    ap.add_argument("--z_coef", type=float, default=0.001)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    a.vocab = tok.get_vocab_size()
    train = Data(os.path.join(a.data, "train.bin"), a.seq)
    val = Data(os.path.join(a.data, "val.bin"), a.seq)
    log("device=%s vocab=%d train tokens=%dM"
        % (DEV, a.vocab, len(train.d) // 1_000_000))
    res = [run(n, a, train, val, tok) for n in a.models.split(",")]
    print()
    print("=== Small MoE vs dense, same active compute (%dM tokens each) ==="
          % (a.tokens // 1_000_000))
    print("model   params  active  val loss  perplexity  tokens/s")
    for r in res:
        print("%-6s %6.1fM %6.1fM   %6.4f   %9.1f  %s"
              % (r["name"], r["total_M"], r["active_M"], r["val_loss"],
                 r["ppl"], "%.0fk" % (r["tok_s"] / 1e3) if r["tok_s"]
                 else "-"))
    for r in res:
        print()
        print("%s sample: %s" % (r["name"], r["sample"].replace("\n", " ")))


if __name__ == "__main__":
    main()
