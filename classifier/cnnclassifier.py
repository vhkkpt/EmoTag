import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnClassifier(nn.Module):
    def __init__(self, embeddings, embedding_dim, num_classes=1, num_filters=50, kernel_sizes=(3, 4, 5)):
        super(CnnClassifier, self).__init__()
        self.num_classes = num_classes
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * num_filters, num_classes),
            nn.Dropout(0.5)
        )

    def forward(self, x):  # x: (batch_size, max_len)
        emb = self.embeddings(x)  # (batch_size, max_len, embedding_dim)
        emb = emb.unsqueeze(1)  # (batch_size, 1, max_len, embedding_dim)

        conv_outs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, max_len-k+1), ...]
        pooled = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]  # [(batch_size, num_filters), ...]
        cat = torch.cat(pooled, dim=1)  # (batch_size, len(kernel_sizes) * num_filters)
        out = self.fc(cat)
        return out
