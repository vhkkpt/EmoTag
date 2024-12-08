import torch.nn as nn


class FnnClassifier(nn.Module):
    def __init__(self, embeddings, embedding_dim, num_classes=1, h_size=64):
        super(FnnClassifier, self).__init__()
        self.num_classes = num_classes
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, h_size),
            nn.ReLU(),
            nn.Linear(h_size, num_classes)
        )
    
    def forward(self, x): # x: (batch_size, max_len)
        emb = self.embeddings(x)
        avg_emb = emb.mean(dim=1)
        out = self.fc(avg_emb)
        return out
