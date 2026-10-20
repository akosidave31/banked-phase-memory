"""
pmlm_b7.py - Track B v7: teach the splitter English structure, not
our 5 relations.
Qwen 1.5B writes phrasings for 40 unrelated "decoy" relations
(mother, pet dog, logo, slogan, ...) and ~30 question openers. The
BIO tagger (pmlm_b6) trains on a mix of normal questions and nested
questions built from decoy + real training phrases. Relations are
still chosen zero-shot by Qwen 0.5B under the schema (as v6).
Guards: generated phrases with a held-out word stem are dropped;
generated openers containing "you know" / "you tell" are dropped, so
the test openers stay unseen. Decoys are cached in decoys.json.
All real phrases come from v1 training families only.
Needs pmlm_b.py, pmlm_b2.py, pmlm_b4.py, pmlm_zs.py, pmlm_b6.py,
relations_v1.json.
"""
import argparse
import collections
import json
import os
import random
import re
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pmlm_b as B
import pmlm_b2 as B2
import pmlm_b4 as B4
import pmlm_b6 as B6

T0 = time.time()
DEV = B.DEV
NR, HMAX = B.NR, B.HMAX
CONDS = [("A", 0), ("B", 1), ("D", 3)]
CAUSES = ["correct", "name", "hop-count", "relation", "memory"]
DECOYS = [(r, "person") for r in [
    "mother", "father", "older sister", "younger brother",
    "grandmother", "uncle", "cousin", "spouse", "eldest child",
    "pet dog", "pet cat", "favorite book", "favorite song",
    "favorite movie", "favorite food", "car", "bicycle", "phone",
    "birthday", "hobby", "doctor", "dentist", "lawyer",
    "piano teacher", "university", "nickname"]] + [
    (r, "company") for r in [
        "logo", "slogan", "main rival", "best-selling product",
        "website", "mascot", "annual revenue", "parent company",
        "advertising agency", "stock ticker", "newest product",
        "biggest customer", "main supplier", "accounting firm"]]
PHRASE_PROMPT = (
    'Write %d different English phrases that mean "the %s of" a %s. '
    'Each phrase must end exactly where the name would go, so that '
    '"<phrase> NAME" reads naturally. Vary the wording and grammar '
    '(for example "the X of", "the X that belongs to", "the X used '
    'by"). One phrase per line, no numbering, no name.')
OPENER_PROMPT = (
    'Write 30 different ways to begin an English question that asks '
    'for a person, a company or a city, such as "What is" or "Name". '
    'Write only the opening words (one to five words), one per line, '
    'no numbering, no question mark.')
REFS = [
    "v6 tagger+qwen (no decoys) B 0.994  0.908  0.784  0.672 | "
    "D 0.734  0.450  0.403  0.734",
    "v6 slots+schema            B 0.892  0.957  0.871  0.831 | "
    "D 0.934  0.931  0.888  0.853",
    "v6 ceiling (true phrases)  B 1.000  0.993  0.980  0.977 | "
    "D 1.000  0.990  0.993  0.977",
    "v6 tagger splitting B, exactly right: 0.971  0.906  0.786  0.651"]


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def held_stems(held):
    stems = set()
    for fams in held.values():
        for f in fams:
            stems.add(f.lower())
            if f.endswith("e"):
                stems.add(f[:-1].lower())
    return stems


def clean_phrase(line):
    s = re.sub(r"^\s*(\d+\s*[.):-]|[-*\u2022])\s*", "", line).strip()
    s = s.strip("\"'` ")
    s = re.sub(r"\s+(name|\[name\]|<name>|x|\.\.\.|\u2026)$", "", s,
               flags=re.I)
    s = s.rstrip(" .,;:!?\"'")
    if not re.fullmatch(r"[A-Za-z][A-Za-z' -]*", s):
        return ""
    if not 2 <= len(s.split()) <= 10:
        return ""
    return s[0].lower() + s[1:]


def clean_opener(line):
    s = re.sub(r"^\s*(\d+\s*[.):-]|[-*\u2022])\s*", "", line).strip()
    s = s.strip("\"'` ").rstrip(" .,;:!?\"'")
    s = re.sub(r"\s*(\.\.\.|\u2026)$", "", s)
    if not re.fullmatch(r"[A-Za-z][A-Za-z' ,-]*", s):
        return ""
    if not 1 <= len(s.split()) <= 6:
        return ""
    return s[0].upper() + s[1:]


@torch.no_grad()
def generate(lm, tok, prompts, max_new, bs=8):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEV)
        g = lm.generate(**enc, max_new_tokens=max_new, do_sample=False,
                        pad_token_id=tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    tok.padding_side = "right"
    return outs


def make_decoys(a, stems, known):
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.gen_lm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.gen_lm, dtype=dtype)
    lm = lm.to(DEV).eval()
    log("generating decoy phrases with %s" % a.gen_lm)
    outs = generate(lm, tok, [PHRASE_PROMPT % (a.n_phr, r, s)
                              for r, s in DECOYS], 400)
    rels, dropped = {}, []
    for (r, _), out in zip(DECOYS, outs):
        keep = []
        for line in out.splitlines():
            s = clean_phrase(line)
            if not s:
                continue
            if any(w.startswith(st) for w in B4.words(s) for st in stems):
                dropped.append(s)
                continue
            if s in known or s in keep:
                continue
            keep.append(s)
        rels[r] = keep
    op_out = generate(lm, tok, [OPENER_PROMPT], 300)[0]
    openers, op_drop = [], []
    for line in op_out.splitlines():
        s = clean_opener(line)
        if not s:
            continue
        low = s.lower()
        if "you know" in low or "you tell" in low or s in B.TE_OPEN:
            op_drop.append(s)
            continue
        if s in B.TR_OPEN or s in openers:
            continue
        openers.append(s)
    del lm
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return dict(gen_lm=a.gen_lm, relations=rels, openers=openers,
                dropped=dropped, openers_dropped=op_drop)


def decoy_q(world, rng, pool, decoys, openers):
    h = rng.randint(1, HMAX)
    opener = rng.choice(openers) if rng.random() < 0.9 else ""
    phrases = []
    for _ in range(h):
        if rng.random() < 0.5:
            phrases.append(rng.choice(rng.choice(decoys)))
        else:
            phrases.append(rng.choice(pool["np"][rng.randrange(NR)]))
    s = rng.choice(world.groups[rng.choice([0, 1])])
    text, segs = opener, []
    for k, ph in enumerate(phrases):
        if text:
            text += " "
        a0 = len(text)
        text += ph
        segs.append((a0, len(text), h - k))
    text += " "
    n0 = len(text)
    text += world.name[s]
    n1 = len(text)
    text += "?"
    return dict(text=text, name=(n0, n1), segs=segs)


def train_tagger2(m, lm, tok, a, rng, pool, decoys, openers):
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    every = max(a.steps // 6, 1)
    world = None
    for step in range(1, a.steps + 1):
        if step == 1 or step % 50 == 0:
            world = B.World(a.n_train, rng)
        qs = [decoy_q(world, rng, pool, decoys, openers)
              if rng.random() < a.decoy_frac
              else B2.train_q(world, rng, pool) for _ in range(a.bs)]
        enc = B.tokenize(tok, qs)
        mask = enc["attention_mask"].bool().to(DEV)
        f = B2.feats(lm, enc, a.k_layers)
        tg = B6.bio_targets(enc, qs).to(DEV).masked_fill(~mask, -100)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gen_lm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--decoys", default="decoys.json")
    ap.add_argument("--n_phr", type=int, default=15)
    ap.add_argument("--decoy_frac", type=float, default=0.5)
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
    stems = held_stems(held)
    v1 = json.load(open(a.data))
    known = {it["text"].strip() for d in v1.values()
             for items in d.values() for it in items}
    if os.path.exists(a.decoys):
        dec = json.load(open(a.decoys))
        log("loaded %s" % a.decoys)
    else:
        dec = make_decoys(a, stems, known)
        json.dump(dec, open(a.decoys, "w"), indent=1)
        log("saved %s" % a.decoys)
    decoys = [v for v in dec["relations"].values() if v]
    openers = list(B.TR_OPEN) + dec["openers"]
    n_dec = sum(len(v) for v in decoys)
    print("decoy phrases kept: %d from %d relations; dropped by stem "
          "guard: %d" % (n_dec, len(decoys), len(dec["dropped"])))
    print("openers: %d generated kept, %d dropped by test-opener guard"
          % (len(dec["openers"]), len(dec["openers_dropped"])))
    for r, v in list(dec["relations"].items())[::4]:
        print("  %-20s %s" % (r, " | ".join(v[:3])))
    print("  openers: %s" % " | ".join(dec["openers"][:12]))
    print("  dropped (guard): %s" % " | ".join(dec["dropped"][:8]))
    print("  openers dropped: %s" % " | ".join(dec["openers_dropped"]))

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
    qwen = B6.QwenRel(lm, tok, dummy)
    acc = {c: [] for c, _ in CONDS}
    cause = [collections.Counter() for _ in range(HMAX)]
    seg = {c: [[0, 0, 0] for _ in range(HMAX)] for c, _ in CONDS}
    examples = []
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        tg = B6.BIOTagger(hid, maxlen=128).to(DEV)
        log("seed %d: training tagger with decoys" % seed)
        train_tagger2(tg, lm, tok, a, random.Random(seed + 1), tr,
                      decoys, openers)
        for c, _ in CONDS:
            per_h = []
            for h in range(HMAX):
                qs = qsets[c][h]
                items = B6.parse_bio(tg, lm, tok, qs, a.k_layers)
                parses = B6.qwen_parses(qwen, items, W, widx)
                recs = B6.answer(mem, addr, W, qs, parses, a)
                per_h.append(recs.count("correct") / len(recs))
                if c == "B":
                    cause[h].update(recs)
                for q, (nm, ph) in zip(qs, items):
                    gp = B6.gold_phrases(q)
                    st = seg[c][h]
                    st[0] += 1
                    st[1] += int(ph == gp)
                    st[2] += int(len(ph) == len(gp))
                    if (c == "B" and seed == 0 and h >= 1
                            and ph != gp and len(examples) < 10):
                        examples.append((gp, ph))
            acc[c].append(per_h)
        log("seed %d: B %s | D %s" % (seed, B2.fmt(acc["B"][-1]),
                                      B2.fmt(acc["D"][-1])))
        del tg
        if DEV == "cuda":
            torch.cuda.empty_cache()

    def mean(v):
        return sum(v) / len(v)

    print()
    print("=== Track B v7: splitter trained on structure (decoys), "
          "Qwen relations (%s, %d seeds) ===" % (a.lm, a.seeds))
    print("decoys: %d phrases / %d relations by %s; openers: %d "
          "generated + %d training"
          % (n_dec, len(decoys), dec.get("gen_lm", "?"),
             len(dec["openers"]), len(B.TR_OPEN)))
    print()
    print("ANSWERS, tagger(decoys)+qwen (mean over seeds)")
    print("cond  1-hop  2-hop  3-hop  4-hop   4-hop by seed")
    for c, _ in CONDS:
        rows = acc[c]
        hm = [mean([r[h] for r in rows]) for h in range(HMAX)]
        print("%-4s  %s   %s" % (c, B2.fmt(hm),
                                 " ".join("%.3f" % r[3] for r in rows)))
    for ref in REFS:
        print("ref: " + ref)
    print()
    print("WHY CONDITION B FAILS (fraction of questions, seeds pooled)")
    print("hops  " + "  ".join("%9s" % c for c in CAUSES))
    for h in range(HMAX):
        tot = max(sum(cause[h].values()), 1)
        print("  %d   %s" % (h + 1, "  ".join(
            "%9.3f" % (cause[h][c] / tot) for c in CAUSES)))
    print()
    print("SPLITTING (seeds pooled): phrases exactly right / hop count "
          "right")
    print("cond  " + "   ".join("%d-hop      " % (h + 1)
                                for h in range(HMAX)))
    for c, _ in CONDS:
        print("%-4s  %s" % (c, "   ".join(
            "%5.3f/%5.3f" % (st[1] / max(st[0], 1), st[2] / max(st[0], 1))
            for st in seg[c])))
    print()
    print("SPLIT MISTAKES, condition B (seed 0; phrases outermost first)")
    for gp, ph in examples:
        print("  true: %s" % " | ".join(gp[::-1]))
        print("  got : %s" % " | ".join(ph[::-1]))
    if not examples:
        print("  none")
    log("done")


if __name__ == "__main__":
    main()
