"""
pmlm_zs.py - Does the frozen LM already know what each phrase means?
Zero-shot test, no training. For every relation phrase in
relations_v1.json, ask the instruct LM:
  Question: "What is <phrase> <name>?"  What is this asking for?
  A) ... E) (the 5 relations, described without any held-out word)
and read the probabilities of the answer letters. Each phrase is
tried with several names; option order is rotated to cancel letter
bias.
  all-5 : choose among all 5 relations
  typed : only relations that fit the name's type (person/company)
Compares held-out phrases with seen ones, and with the trained
parser's per-phrase accuracy from earlier runs (v1 training data).
Needs pmlm_b.py, pmlm_b2.py, pmlm_b4.py, relations_v1.json.
"""
import argparse
import collections
import random
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pmlm_b as B
import pmlm_b2 as B2
import pmlm_b4 as B4

T0 = time.time()
DEV = B.DEV
NR = B.NR
REL = B2.REL_NAMES
DESC = ["the company that employs this person",
        "the city where this person resides",
        "the city where this company has its headquarters",
        "a friend of this person",
        "the person who heads this company"]
LETTERS = "ABCDE"
PARSER_V1 = {
    "the workplace of": 0.997, "the place of work of": 0.929,
    "the company that pays": 0.947, "the city inhabited by": 0.996,
    "the locality inhabited by": 1.000, "the town lived in by": 0.996,
    "the location of": 0.986, "the base of": 0.218,
    "the base city of": 0.984, "the seat of": 0.568,
    "the home base of": 0.828, "the operating base of": 0.907,
    "the corporate home of": 0.526,
    "the city that serves as the base of": 0.949,
    "the location of the main office of": 1.000,
    "a buddy of": 0.984, "a mate of": 0.989,
    "one of the buddies of": 0.997, "a person close to": 0.916,
    "the boss of": 1.000, "the chief of": 0.979,
    "the chief executive of": 0.995,
    "the chief executive officer of": 0.995, "the top boss of": 1.000,
    "the person steering": 1.000, "the individual who directs": 0.983}


def log(m):
    print("[%6.0fs] %s" % (time.time() - T0, m), flush=True)


def label(name):
    m = re.search(r"(\d+(?:\.\d+)?)B", name)
    return (m.group(1) + "B") if m else name.split("/")[-1][:8]


def opts_for(r, mode):
    if mode == "all5":
        return list(range(NR))
    return [x for x in range(NR) if B.DOM[x] == B.DOM[r]]


def prompt(tok, np_text, name, opts):
    lines = ["%s) %s" % (LETTERS[i], DESC[r]) for i, r in enumerate(opts)]
    msg = ('Question: "What is %s %s?"\nWhat is this question asking '
           'for?\n%s\nAnswer with the letter only.'
           % (np_text, name, "\n".join(lines)))
    return tok.apply_chat_template([{"role": "user", "content": msg}],
                                   tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def score(lm, tok, texts, nopts, bs):
    ids = [tok.encode(c, add_special_tokens=False) for c in LETTERS]
    assert all(len(x) == 1 for x in ids), ids
    ids = torch.tensor([x[0] for x in ids], device=DEV)
    tok.padding_side = "right"
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False)
        am = enc["attention_mask"].to(DEV)
        hs = lm.model(input_ids=enc["input_ids"].to(DEV),
                      attention_mask=am).last_hidden_state
        last = am.sum(1) - 1
        h = hs[torch.arange(hs.shape[0], device=DEV), last]
        lg = lm.lm_head(h).float()[:, ids].cpu()
        for row, n in zip(lg, nopts[i:i + bs]):
            out.append(row[:n].log_softmax(0).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lms", default="Qwen/Qwen2.5-0.5B-Instruct,"
                                     "Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="relations_v1.json")
    ap.add_argument("--names", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    tr, te, held = B2.load_split(a.data)
    hl = B4.held_list(a.data, te)
    phrases = ([(r, t, "held") for r, _, t in hl]
               + [(r, t, "seen") for r in range(NR) for t in tr["np"][r]])
    W = B.World(64, random.Random(7))
    names = [[W.name[s] for s in W.groups[t][:a.names]] for t in (0, 1)]
    jobs = []
    for r, t, _ in phrases:
        for ni in range(a.names):
            for mode in ("all5", "typed"):
                opts = opts_for(r, mode)
                for k in range(len(opts)):
                    jobs.append(((r, t, ni, mode), opts[k:] + opts[:k]))
    log("%d phrases (%d held-out), %d prompts per LM"
        % (len(phrases), len(hl), len(jobs)))

    res, labs = {}, []
    for lm_name in a.lms.split(","):
        lab = label(lm_name)
        labs.append(lab)
        dtype = torch.float16 if DEV == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(lm_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        lm = AutoModelForCausalLM.from_pretrained(lm_name, dtype=dtype)
        lm = lm.to(DEV).eval()
        texts = [prompt(tok, t, names[B.DOM[r]][ni], rot)
                 for (r, t, ni, mode), rot in jobs]
        log("%s: scoring %d prompts" % (lm_name, len(texts)))
        lps = score(lm, tok, texts, [len(rot) for _, rot in jobs], a.bs)
        sc = collections.defaultdict(lambda: [0.0] * NR)
        for (key, rot), lp in zip(jobs, lps):
            for rel, v in zip(rot, lp):
                sc[key][rel] += v
        per = collections.defaultdict(
            lambda: [0, 0, collections.Counter()])
        for (r, t, ni, mode), v in sc.items():
            pred = max(opts_for(r, mode), key=lambda x: v[x])
            st = per[(r, t, mode)]
            st[1] += 1
            if pred == r:
                st[0] += 1
            else:
                st[2][pred] += 1
        res[lab] = per
        log("%s done" % lab)
        del lm
        if DEV == "cuda":
            torch.cuda.empty_cache()

    def acc(lab, r, t, mode):
        st = res[lab][(r, t, mode)]
        return st[0] / max(st[1], 1)

    def top_wrong(lab, r, t, mode):
        c = res[lab][(r, t, mode)][2]
        return REL[c.most_common(1)[0][0]] if c else "-"

    print()
    print("=== Zero-shot: does the frozen LM already know what each "
          "phrase means? ===")
    print("no training; each phrase x %d names; option order rotated" %
          a.names)
    print("all-5 = pick among 5 relations; typed = only relations that "
          "fit the name's type")
    print()
    print("SUMMARY (mean accuracy over phrases; relation columns are "
          "all-5)")
    print("%-5s %-5s %4s  %5s  %5s   %s"
          % ("LM", "kind", "n", "all-5", "typed",
             " ".join("%-10s" % x for x in REL)))
    for lab in labs:
        for kind in ("held", "seen"):
            ps = [(r, t) for r, t, k in phrases if k == kind]
            a5 = sum(acc(lab, r, t, "all5") for r, t in ps) / len(ps)
            ty = sum(acc(lab, r, t, "typed") for r, t in ps) / len(ps)
            byr = []
            for rr in range(NR):
                q = [acc(lab, r, t, "all5") for r, t in ps if r == rr]
                byr.append("%-10s" % (("%.3f" % (sum(q) / len(q)))
                                      if q else "-"))
            print("%-5s %-5s %4d  %5.3f  %5.3f   %s"
                  % (lab, kind, len(ps), a5, ty, " ".join(byr)))
    print()
    print("HELD-OUT PHRASES (accuracy over names; parser = trained "
          "parser on v1 data, earlier runs)")
    head = "".join("  %-11s" % (lab + " a5/typ") for lab in labs)
    print("%-10s %-35s %6s%s  LM wrong pick (all-5)"
          % ("relation", "phrase", "parser", head))
    for r, fam, t in hl:
        p = PARSER_V1.get(t)
        cols = "".join("  %5.3f/%5.3f" % (acc(lab, r, t, "all5"),
                                          acc(lab, r, t, "typed"))
                       for lab in labs)
        wrong = " / ".join(top_wrong(lab, r, t, "all5") for lab in labs)
        print("%-10s %-35s %6s%s  %s"
              % (REL[r], t, ("%.3f" % p) if p is not None else "-",
                 cols, wrong))
    print()
    print("SEEN PHRASES the LM misreads (all-5 below 0.75) - checks "
          "the prompt itself")
    for lab in labs:
        seen = [(r, t) for r, t, k in phrases if k == "seen"]
        bad = [(r, t) for r, t in seen if acc(lab, r, t, "all5") < 0.75]
        print("%s: %d of %d" % (lab, len(bad), len(seen)))
        for r, t in bad:
            print("   %-10s %-40s %.3f -> %s"
                  % (REL[r], t, acc(lab, r, t, "all5"),
                     top_wrong(lab, r, t, "all5")))
    log("done")


if __name__ == "__main__":
    main()
