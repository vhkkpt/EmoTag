from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class TextDataset(Dataset):
    def __init__(self, file_path, tokenize_fn):
        df = pd.read_csv(file_path)
        self.x = []
        self.y = []
        for _, row in df.iterrows():
            self.x.append(torch.tensor(tokenize_fn(row.content)))
            self.y.append(row.label)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def collate_fn(batch, device):
    b_x, b_y = [], []
    max_len = np.max([len(x) for x, _ in batch])
    for x, y in batch:
        b_x.append(torch.concat([x, torch.zeros(max_len - len(x))]))
        b_y.append(y)
    return torch.stack(b_x).int().to(device), torch.stack(b_y).to(device)


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs.squeeze(), y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_accurary = 0
    total_numel = 0
    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y.float())
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            total_accurary += (predictions == y.float()).sum().item()
            total_numel += y.numel()
    total_loss /= len(dataloader)
    total_accurary /= total_numel
    return total_loss, total_accurary
