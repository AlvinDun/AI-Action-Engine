import argparse, os, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv

from app.preprocess import build_vocab, encode, save_vocab
from app.model import TextCNN

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)

class LogDataset(Dataset):
    def __init__(self, rows, vocab, max_len, labels):
        self.rows = rows
        self.vocab = vocab
        self.max_len = max_len
        self.labels = labels
        self.label_to_idx = {l:i for i,l in enumerate(labels)}

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        x = encode(text, self.vocab, self.max_len)
        y = self.label_to_idx[label]
        return x, y

def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line in r:
            rows.append((line["text"], line["label"]))
    return rows

def split(rows, val_ratio=0.2, seed=42):
    random.Random(seed).shuffle(rows)
    n = int(len(rows)*(1-val_ratio))
    return rows[:n], rows[n:]

def train(args):
    rows = load_data(DATA)
    labels = sorted(list({lab for _, lab in rows}))
    texts = [t for t,_ in rows]
    vocab = build_vocab(texts, min_freq=1, max_size=20000)
    save_vocab(vocab, str(ART / "vocab.json"))
    with open(ART / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)

    tr, va = split(rows, val_ratio=0.3, seed=123)
    tr_ds = LogDataset(tr, vocab, args.max_len, labels)
    va_ds = LogDataset(va, vocab, args.max_len, labels)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size)

    model = TextCNN(vocab_size=len(vocab), embed_dim=64, num_classes=len(labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_va = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x,y in tr_dl:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))
        tr_acc = correct/total if total else 0.0

        model.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for x,y in va_dl:
                logits = model(x)
                pred = logits.argmax(dim=-1)
                vcorrect += int((pred == y).sum().item())
                vtotal += int(x.size(0))
        va_acc = vcorrect/vtotal if vtotal else 0.0

        print(f"epoch {epoch} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")
        if va_acc >= best_va:
            best_va = va_acc
            torch.save(model.state_dict(), ART / "model.pt")
    print(f"best_val_acc={best_va:.3f} | artifacts saved to {ART}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    args = p.parse_args()
    train(args)
