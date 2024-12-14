from glove import GloVe
from classifier.fnnclassifier import FnnClassifier
from classifier.cnnclassifier import CnnClassifier
from classifier.transformerclassifier import TransformerClassifier
from classifier.bertclassifier import BertClassifier
from dataset import textdataset
from dataset import berttextdataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys


GLOVE_EMBEDDING_DIM = 200
BATCH_SIZE = 8
BERT_MAX_LENGTH = 256


def main(model_type):
    torch.manual_seed(19260817)
    np.random.seed(19260817)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("Chosen model type:", model_type)

    if model_type in ['FNN', 'CNN', 'Transformer']:
        print("Loading GloVe embeddings...")
        glove = GloVe(f'glove.6B.{GLOVE_EMBEDDING_DIM}d.txt', GLOVE_EMBEDDING_DIM)
        train_set = textdataset.TextDataset('data/train.csv', lambda text: glove.tokenize_fn(text))
        val_set = textdataset.TextDataset('data/val.csv', lambda text: glove.tokenize_fn(text))
        test_set = textdataset.TextDataset('data/test.csv', lambda text: glove.tokenize_fn(text))
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda batch: textdataset.collate_fn(batch, device))
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=lambda batch: textdataset.collate_fn(batch, device))
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=lambda batch: textdataset.collate_fn(batch, device))
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_set = berttextdataset.BertTextDataset('data/train.csv', tokenizer, max_length=BERT_MAX_LENGTH)
        val_set = berttextdataset.BertTextDataset('data/val.csv', tokenizer, max_length=BERT_MAX_LENGTH)
        test_set = berttextdataset.BertTextDataset('data/test.csv', tokenizer, max_length=BERT_MAX_LENGTH)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    if model_type == 'FNN':
        model = FnnClassifier(glove.embeddings, glove.embedding_dim).to(device)
        lr = 5e-4
        num_epochs = 50
    elif model_type == 'CNN':
        model = CnnClassifier(glove.embeddings, glove.embedding_dim).to(device)
        lr = 5e-4
        num_epochs = 50
    elif model_type == 'Transformer':
        model = TransformerClassifier(glove.embeddings, glove.embedding_dim).to(device)
        lr = 5e-5
        num_epochs = 50
    else:
        model = BertClassifier().to(device)
        lr = 2e-5
        num_epochs = 10

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0.0

    for epoch in range(num_epochs):
        if model_type in ['FNN', 'CNN', 'Transformer']:
            train_loss = textdataset.train_epoch(model, train_loader, optimizer, criterion)
            test_loss, test_acc = textdataset.evaluate(model, test_loader, criterion)
        else:
            train_loss = berttextdataset.train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = berttextdataset.evaluate(model, test_loader, criterion, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(f'Epoch {epoch+1}: train_loss {train_loss:.4f}, test_loss {test_loss:.4f}, test_acc {test_acc:.4f}')

    print(f'best_test_acc {best_test_acc:.4f}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify the model in the startup parameters [ FNN | CNN | Transformer | BERT ]")
        exit(0)
    assert sys.argv[1] in ['FNN', 'CNN', 'Transformer', 'BERT']
    main(sys.argv[1])
