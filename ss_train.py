"""Part 1 (Easy): 3-state secondary structure prediction (H = helix, E = strand, C = coil).

Two models are compared:
  baseline : trainable amino-acid embedding + BiLSTM  (no pretrained knowledge)
  esm      : frozen ESM-2 embeddings + BiLSTM         (transfer learning)

Data: either a CSV with columns `sequence,ss3` (--csv path) or a Hugging Face dataset.
The default HF dataset name and column names are from memory -- run once and read the printed
column names, then fix --seq_col / --ss_col / --dataset if needed.

Example:
  python ss_train.py --model baseline
  python ss_train.py --model esm --max_train 3000
"""
import argparse
import csv

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from common import embed, get_device, load_esm

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i + 1 for i, a in enumerate(AA)}  # 0 = padding / unknown
SS2I = {"H": 0, "E": 1, "C": 2}


def load_pairs(args):
    """Return dict split -> list of (sequence, ss3_string)."""
    if args.csv:
        rows = list(csv.DictReader(open(args.csv)))
        pairs = [(r["sequence"], r["ss3"]) for r in rows]
        n = len(pairs)
        return {"train": pairs[: int(0.8 * n)], "val": pairs[int(0.8 * n): int(0.9 * n)],
                "test": pairs[int(0.9 * n):]}
    from datasets import load_dataset
    ds = load_dataset(args.dataset)
    print("Splits:", list(ds.keys()), "| columns:", ds[list(ds.keys())[0]].column_names)
    out = {}
    for name, split in ds.items():
        out[name] = [(r[args.seq_col], r[args.ss_col]) for r in split]
    # map whatever the dataset calls its splits to train/val/test
    train = out.get("train")
    val = out.get("validation") or out.get("val")
    test = out.get(args.test_split) or out.get("test")
    return {"train": train, "val": val, "test": test}


def clean(pairs, max_len):
    keep = []
    for s, y in pairs:
        if len(s) == len(y) and len(s) <= max_len and set(y) <= set(SS2I):
            keep.append((s, y))
    return keep


class Net(nn.Module):
    def __init__(self, mode, d_esm=320, hidden=128):
        super().__init__()
        self.mode = mode
        if mode == "baseline":
            self.emb = nn.Embedding(len(AA) + 1, 64, padding_idx=0)
            d_in = 64
        else:
            d_in = d_esm
        self.drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(d_in, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        if self.mode == "baseline":
            x = self.emb(x)
        x = self.drop(x)
        h, _ = self.lstm(x)
        return self.out(self.drop(h))


def make_batches(items, bs, shuffle):
    idx = torch.randperm(len(items)).tolist() if shuffle else list(range(len(items)))
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i: i + bs]]


def collate(batch, device):
    xs, ys = zip(*batch)
    x = pad_sequence(xs, batch_first=True).to(device)
    y = pad_sequence(ys, batch_first=True, padding_value=-100).to(device)
    return x, y


def q3(model, items, device, bs=32):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in make_batches(items, bs, False):
            x, y = collate(batch, device)
            pred = model(x).argmax(-1)
            m = y != -100
            correct += (pred[m] == y[m]).sum().item()
            total += m.sum().item()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "esm"], default="esm")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dataset", default="proteinea/secondary_structure_prediction")
    ap.add_argument("--seq_col", default="input")
    ap.add_argument("--ss_col", default="dssp3")
    ap.add_argument("--test_split", default="CB513")
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--max_train", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = get_device()
    data = {k: clean(v, args.max_len) for k, v in load_pairs(args).items() if v}
    data["train"] = data["train"][: args.max_train]
    print({k: len(v) for k, v in data.items()})

    if args.model == "esm":
        tok, esm, _ = load_esm(device=device)
        d_esm = esm.config.hidden_size

    def to_items(pairs):
        items = []
        for s, y in pairs:
            if args.model == "esm":
                x = embed(s, tok, esm, device)
            else:
                x = torch.tensor([AA2I.get(a, 0) for a in s])
            items.append((x, torch.tensor([SS2I[c] for c in y])))
        return items

    sets = {k: to_items(v) for k, v in data.items()}
    model = Net(args.model, d_esm=d_esm if args.model == "esm" else 320).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best = 0
    for ep in range(args.epochs):
        model.train()
        tot = 0
        for batch in make_batches(sets["train"], args.bs, True):
            x, y = collate(batch, device)
            loss = loss_fn(model(x).transpose(1, 2), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        val = q3(model, sets["val"], device) if "val" in sets else float("nan")
        if val >= best:
            best = val
            torch.save(model.state_dict(), f"ss_{args.model}.pt")
        print(f"epoch {ep + 1:02d} loss {tot:.3f} val Q3 {val:.4f}")

    model.load_state_dict(torch.load(f"ss_{args.model}.pt"))
    if "test" in sets:
        print(f"TEST Q3 ({args.model}): {q3(model, sets['test'], device):.4f}")


if __name__ == "__main__":
    main()
