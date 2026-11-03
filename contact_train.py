"""Part 2 (Medium): contact map prediction from sequence using frozen ESM-2 embeddings + a 2D ResNet.

Contact = C-beta distance < 8 A. We ignore pairs with sequence separation < 6.
Metric: precision of the top L/5 predicted LONG-range contacts (|i-j| >= 24), the standard one.

Example:
  python contact_prep.py chains.txt
  python contact_train.py --data data/contacts --epochs 20
"""
import argparse
import glob
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import embed, get_device, load_esm

CONTACT_CUTOFF = 8.0
MIN_SEP = 6
LONG_SEP = 24


class ResBlock(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.n1 = nn.BatchNorm2d(c)
        self.n2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class ContactNet(nn.Module):
    def __init__(self, d_esm=320, d_proj=48, c=64, n_blocks=6):
        super().__init__()
        self.proj = nn.Linear(d_esm, d_proj)
        self.inp = nn.Conv2d(3 * d_proj + 1, c, 1)
        self.blocks = nn.Sequential(*[ResBlock(c, d) for d in [1, 2, 4, 8, 1, 2][:n_blocks]])
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, emb):  # emb: (L, D)
        a = self.proj(emb)  # (L, P)
        L = a.shape[0]
        ai = a[:, None].expand(L, L, -1)
        aj = a[None].expand(L, L, -1)
        pos = torch.arange(L, device=emb.device)
        sep = (pos[:, None] - pos[None]).abs().float()[..., None]
        sep = torch.log1p(sep) / 6.0  # relative position feature
        x = torch.cat([ai, aj, ai * aj, sep], dim=-1).permute(2, 0, 1)[None]  # (1, C, L, L)
        logits = self.out(self.blocks(self.inp(x)))[0, 0]
        return (logits + logits.T) / 2  # contact maps are symmetric


def pair_mask(L, device, min_sep=MIN_SEP):
    pos = torch.arange(L, device=device)
    return (pos[:, None] - pos[None]).abs() >= min_sep


def top_l5_precision(logits, contacts, valid):
    L = logits.shape[0]
    mask = pair_mask(L, logits.device, LONG_SEP) & valid
    mask = torch.triu(mask, diagonal=1)
    scores = logits[mask]
    truth = contacts[mask]
    k = max(L // 5, 1)
    if scores.numel() < k:
        return float("nan")
    top = scores.topk(k).indices
    return truth[top].float().mean().item()


def load_items(folder, tok, esm, device):
    items = []
    for f in sorted(glob.glob(f"{folder}/*.npz")):
        z = np.load(f)
        seq, dist = str(z["seq"]), torch.tensor(z["dist"])
        valid = ~torch.isnan(dist)
        contacts = (torch.nan_to_num(dist, nan=1e3) < CONTACT_CUTOFF)
        items.append((embed(seq, tok, esm, device), contacts, valid))
    return items


def evaluate(model, items, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for emb, contacts, valid in items:
            logits = model(emb.to(device)).cpu()
            p = top_l5_precision(logits, contacts, valid)
            if p == p:
                scores.append(p)
    return float(np.mean(scores)) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/contacts")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    tok, esm, _ = load_esm(device=device)
    items = load_items(args.data, tok, esm, device)
    # WARNING: a random split is only OK if your chain list was already clustered at <=30% identity.
    random.seed(0)
    random.shuffle(items)
    n = len(items)
    train, val, test = items[: int(0.8 * n)], items[int(0.8 * n): int(0.9 * n)], items[int(0.9 * n):]
    print(f"train {len(train)} val {len(val)} test {len(test)}")

    model = ContactNet(d_esm=esm.config.hidden_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = -1
    for ep in range(args.epochs):
        model.train()
        random.shuffle(train)
        tot = 0
        for i, (emb, contacts, valid) in enumerate(train):
            logits = model(emb.to(device))
            mask = (pair_mask(logits.shape[0], device) & valid.to(device))
            # contacts are rare (~2-3% of pairs) -> up-weight positives
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], contacts.to(device)[mask].float(),
                pos_weight=torch.tensor(10.0, device=device))
            (loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
            tot += loss.item()
        v = evaluate(model, val, device)
        if v > best:
            best = v
            torch.save(model.state_dict(), "contact_net.pt")
        print(f"epoch {ep + 1:02d} loss {tot / max(len(train), 1):.4f} val P@L/5 (long-range) {v:.4f}")

    model.load_state_dict(torch.load("contact_net.pt"))
    print(f"TEST P@L/5 (long-range): {evaluate(model, test, device):.4f}")


if __name__ == "__main__":
    main()
