import numpy as np
import torch
import re


class GloVe():
    def __init__(self, file_path, embedding_dim):
        stoi = {}
        embeddings = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                split_line = line.split()
                word = split_line[0]
                vector = np.array(split_line[1:], dtype=np.float32)
                if len(vector) != embedding_dim:
                    raise ValueError(f"Embedding dimension mismatch at line {idx+1}. Expected {embedding_dim}, got {len(vector)}")
                stoi[word] = idx
                embeddings.append(vector)
        embeddings = np.array(embeddings)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.stoi = stoi
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim

    def tokenize_fn(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = text.strip()
        oov_i = len(self.stoi) - 1
        return [self.stoi.get(w, oov_i) for w in text.split()]
