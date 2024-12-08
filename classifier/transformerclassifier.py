import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, embeddings, embedding_dim, num_classes=1, 
                 h_size=128, num_heads=4, num_layers=1, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.num_classes = num_classes

        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout,
            dim_feedforward=embedding_dim * 2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, h_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_size, num_classes)
        )

    def forward(self, x):
        emb = self.embeddings(x)
        emb = emb.permute(1, 0, 2)
        transformer_out = self.transformer(emb)
        transformer_out = transformer_out.permute(1, 0, 2)
        pooled = transformer_out.mean(dim=1)
        out = self.fc(pooled)

        return out
