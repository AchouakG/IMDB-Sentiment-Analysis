import numpy as np
from collections import Counter

def build_vocab(texts, max_words=10000):
    counts = Counter(w for text in texts for w in text.split())
    vocab  = {w: i+1 for i, (w, _) in enumerate(counts.most_common(max_words))}
    return vocab

def encode_and_pad(texts, vocab, max_len=200):
    seqs = [[vocab.get(w, 0) for w in text.split()] for text in texts]
    out  = np.zeros((len(seqs), max_len), dtype=np.int32)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        out[i, :len(s)] = s
    return out