from typing import List, Dict, Tuple
import re
import json
import torch

def tokenize(text: str) -> List[str]:
    # simple whitespace + punctuation split
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9\.\-_]+", text)
    return tokens

def build_vocab(texts: List[str], min_freq: int = 1, max_size: int = 20000) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    # reserve 0:PAD, 1:UNK
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        if len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab

def encode(text: str, vocab: Dict[str, int], max_len: int = 128) -> torch.Tensor:
    toks = tokenize(text)
    ids = [vocab.get(tok, 1) for tok in toks]  # 1 = UNK
    if len(ids) < max_len:
        ids = ids + [0]*(max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

def save_vocab(vocab: Dict[str,int], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

def load_vocab(path: str) -> Dict[str,int]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
