from torch.utils.data import Dataset
import torch
import pandas as pd


class BertTextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.texts = self.data['content'].astype(str).tolist()
        self.labels = self.data['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze(0) # shape: (max_length)
        attention_mask = encoding['attention_mask'].squeeze(0) # shape: (max_length)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.float)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.sigmoid(outputs) > 0.5
            total_accuracy += (predictions.float() == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / total_samples
    return avg_loss, avg_accuracy
