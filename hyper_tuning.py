from itertools import product
from sklearn.metrics import accuracy_score
import torch
from glove import GloVe
from classifier.fnnclassifier import FnnClassifier
from classifier.cnnclassifier import CnnClassifier
from classifier.transformerclassifier import TransformerClassifier
from classifier.bertclassifier import BertClassifier
from dataset import textdataset
from dataset import berttextdataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import copy
import numpy as np

# Global variables
GLOVE_EMBEDDING_DIM = 200
BERT_MAX_LENGTH = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
glove = GloVe(f'glove.6B.{GLOVE_EMBEDDING_DIM}d.txt', GLOVE_EMBEDDING_DIM)

# Prepare datasets and data loaders
train_set = textdataset.TextDataset('data/train.csv', lambda text: glove.tokenize_fn(text))
val_set = textdataset.TextDataset('data/val.csv', lambda text: glove.tokenize_fn(text))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_train_set = berttextdataset.BertTextDataset('data/train.csv', tokenizer, max_length=BERT_MAX_LENGTH)
bert_val_set = berttextdataset.BertTextDataset('data/val.csv', tokenizer, max_length=BERT_MAX_LENGTH)

# Define training and evaluation function
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    # Store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        # Training loop (across all batches)
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch[0].to(device), batch[1].to(device).squeeze()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Calculate training loss and accuracy
        train_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Evaluate on validation dataset
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device).squeeze()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Append to metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print epoch metrics
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Find minimum value for val_losses
    min_val_loss = min(val_losses)
    min_val_loss_index = val_losses.index(min_val_loss)
    min_val_loss_epoch = min_val_loss_index + 1

    # Find maximum value for val_accuracies
    max_val_acc = max(val_accuracies)
    max_val_acc_index = val_accuracies.index(max_val_acc)
    max_val_acc_epoch = max_val_acc_index + 1
    val_loss_at_max_acc = val_losses[max_val_acc_index]

    print(f"Minimum validation loss: {min_val_loss:.4f} at epoch {min_val_loss_epoch}")
    print(f"Maximum validation accuracy: {max_val_acc:.4f} at epoch {max_val_acc_epoch}")

    return max_val_acc_epoch, max_val_acc, val_loss_at_max_acc, train_losses, val_losses, train_accuracies, val_accuracies

def bert_train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    # Store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
 
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
 
        # Training loop (across all batches)
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
 
        # Calculate training loss and accuracy
        train_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
 
        # Evaluate on validation dataset
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(input_ids, attention_mask).squeeze()
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
 
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
 
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
 
        # Append to metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
 
        # Print epoch metrics
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
   
    # Find minimum value for val_losses
    min_val_loss = min(val_losses)
    min_val_loss_index = val_losses.index(min_val_loss)
    min_val_loss_epoch = min_val_loss_index + 1
 
    # Find maximum value for val_accuracies
    max_val_acc = max(val_accuracies)
    max_val_acc_index = val_accuracies.index(max_val_acc)
    max_val_acc_epoch = max_val_acc_index + 1
    val_loss_at_max_acc = val_losses[max_val_acc_index]
 
    print(f"Minimum validation loss: {min_val_loss:.4f} at epoch {min_val_loss_epoch}")
    print(f"Maximum validation accuracy: {max_val_acc:.4f} at epoch {max_val_acc_epoch}")
 
    return max_val_acc_epoch, max_val_acc, val_loss_at_max_acc, train_losses, val_losses, train_accuracies, val_accuracies


# Define hyperparameter tuning functions for all 4 models

def fnn_tuning(hyperparams):
    best_params = None
    best_loss = float('inf')
    best_acc = 0
    best_train_losses = []
    best_val_losses = []
    best_train_accuracies = []
    best_val_accuracies = []

    for params in product(*hyperparams.values()):
        # Unpack hyperparameters
        param_dict = dict(zip(hyperparams.keys(), params))

        # Prepare data loaders using batch size from hyperparameters
        train_loader = DataLoader(train_set, batch_size=param_dict['batch_size'], shuffle=True,
                                  collate_fn=lambda batch: textdataset.collate_fn(batch, device))
        val_loader = DataLoader(val_set, batch_size=param_dict['batch_size'], shuffle=False,
                                collate_fn=lambda batch: textdataset.collate_fn(batch, device))

        # Create model
        model = FnnClassifier(glove.embeddings, GLOVE_EMBEDDING_DIM,
                              num_classes=param_dict['num_classes'], 
                              h_size=param_dict['h_size']).to(device)

        optimizer = optim.Adam(model.parameters(), lr=param_dict['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()

        # Train the model and evaluate on validation dataset
        max_val_acc_epoch, max_val_acc, val_loss_at_max_acc, train_losses, val_losses, \
        train_accuracies, val_accuracies = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, param_dict['num_epochs'])

        # Update best parameters based on validation loss and accuracy
        if max_val_acc > best_acc or (max_val_acc == best_acc and val_loss_at_max_acc < best_loss):
            best_loss = val_loss_at_max_acc
            best_acc = max_val_acc
            best_params = copy.deepcopy(param_dict)
            best_params['num_epochs'] = max_val_acc_epoch
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_train_accuracies = train_accuracies
            best_val_accuracies = val_accuracies

    # Plotting the training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_train_losses, label='Training Loss')
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Best Hyperparameter Training and Validation Loss')

    # Plotting the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_train_accuracies, label='Training Accuracy')
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Best Hyperparameter Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()       

    return best_params, best_loss, best_acc

def cnn_tuning(hyperparams):
    best_params = None
    best_loss = float('inf')
    best_acc = 0

    for params in product(*hyperparams.values()):
        # Unpack hyperparameters
        param_dict = dict(zip(hyperparams.keys(), params))

        # Prepare data loaders using batch size from hyperparameters
        train_loader = DataLoader(train_set, batch_size=param_dict['batch_size'], shuffle=True,
                                  collate_fn=lambda batch: textdataset.collate_fn(batch, device))
        val_loader = DataLoader(val_set, batch_size=param_dict['batch_size'], shuffle=False,
                                collate_fn=lambda batch: textdataset.collate_fn(batch, device))

        # Create model
        model = CnnClassifier(glove.embeddings, GLOVE_EMBEDDING_DIM,
                              num_classes=param_dict['num_classes'], 
                              num_filters=param_dict['num_filters'], 
                              kernel_sizes=param_dict['kernel_sizes']).to(device)

        optimizer = optim.Adam(model.parameters(), lr=param_dict['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()

        # Train the model and evaluate on validation dataset
        max_val_acc_epoch, max_val_acc, val_loss_at_max_acc, train_losses, val_losses, \
        train_accuracies, val_accuracies = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, param_dict['num_epochs'])

        # Update best parameters based on validation loss and accuracy
        if max_val_acc > best_acc or (max_val_acc == best_acc and val_loss_at_max_acc < best_loss):
            best_loss = val_loss_at_max_acc
            best_acc = max_val_acc
            best_params = copy.deepcopy(param_dict)
            best_params['num_epochs'] = max_val_acc_epoch
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_train_accuracies = train_accuracies
            best_val_accuracies = val_accuracies

    # Plotting the training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_train_losses, label='Training Loss')
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Best Hyperparameter Training and Validation Loss')

    # Plotting the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_train_accuracies, label='Training Accuracy')
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Best Hyperparameter Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()

    return best_params, best_loss, best_acc

def transformer_tuning(hyperparams):
    best_params = None
    best_loss = float('inf')
    best_acc = 0

    for params in product(*hyperparams.values()):
        # Unpack hyperparameters
        param_dict = dict(zip(hyperparams.keys(), params))

        # Prepare data loaders using batch size from hyperparameters
        train_loader = DataLoader(train_set, batch_size=param_dict['batch_size'], shuffle=True,
                                  collate_fn=lambda batch: textdataset.collate_fn(batch, device))
        val_loader = DataLoader(val_set, batch_size=param_dict['batch_size'], shuffle=False,
                                collate_fn=lambda batch: textdataset.collate_fn(batch, device))

        # Create model
        model = TransformerClassifier(glove.embeddings, GLOVE_EMBEDDING_DIM,
                                      num_classes=param_dict['num_classes'], 
                                      h_size=param_dict['h_size'],
                                      num_heads=param_dict['num_heads'], 
                                      num_layers=param_dict['num_layers'], 
                                      dropout=param_dict['dropout']).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=param_dict['learning_rate'], weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        # Train the model and evaluate on validation dataset
        max_val_acc_epoch, max_val_acc, val_loss_at_max_acc, train_losses, val_losses, \
        train_accuracies, val_accuracies = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, param_dict['num_epochs'])

        # Update best parameters based on validation loss and accuracy
        if max_val_acc > best_acc or (max_val_acc == best_acc and val_loss_at_max_acc < best_loss):
            best_loss = val_loss_at_max_acc
            best_acc = max_val_acc
            best_params = copy.deepcopy(param_dict)
            best_params['num_epochs'] = max_val_acc_epoch
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_train_accuracies = train_accuracies
            best_val_accuracies = val_accuracies

    # Plotting the training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_train_losses, label='Training Loss')
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Best Hyperparameter Training and Validation Loss')

    # Plotting the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_train_accuracies, label='Training Accuracy')
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Best Hyperparameter Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()

    return best_params, best_loss, best_acc

def bert_tuning(hyperparams):
    best_params = None
    best_loss = float('inf')
    best_acc = 0

    for params in product(*hyperparams.values()):
        # Unpack hyperparameters
        param_dict = dict(zip(hyperparams.keys(), params))

        # Prepare data loaders using batch size from hyperparameters
        bert_train_loader = DataLoader(bert_train_set, batch_size=param_dict['batch_size'], shuffle=True)
        bert_val_loader = DataLoader(bert_val_set, batch_size=param_dict['batch_size'], shuffle=False)

        # Create model
        model = BertClassifier(bert_model_name='bert-base-uncased', 
                               num_classes=param_dict['num_classes'], 
                               dropout=param_dict['dropout']).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=param_dict['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()

        # Train the model and evaluate on validation dataset
        max_val_acc_epoch, max_val_acc, val_loss_at_max_acc, train_losses, val_losses, \
        train_accuracies, val_accuracies = bert_train_and_evaluate(model, bert_train_loader, bert_val_loader, optimizer, criterion, param_dict['num_epochs'])

        # Update best parameters based on validation loss and accuracy
        if max_val_acc > best_acc or (max_val_acc == best_acc and val_loss_at_max_acc < best_loss):
            best_loss = val_loss_at_max_acc
            best_acc = max_val_acc
            best_params = copy.deepcopy(param_dict)
            best_params['num_epochs'] = max_val_acc_epoch
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_train_accuracies = train_accuracies
            best_val_accuracies = val_accuracies

    # Plotting the training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_train_losses, label='Training Loss')
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Best Hyperparameter Training and Validation Loss')

    # Plotting the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_train_accuracies, label='Training Accuracy')
    plt.plot(range(1, param_dict['num_epochs'] + 1), best_val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Best Hyperparameter Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()

    return best_params, best_loss, best_acc


# Hyperparameter tuning for all models
def main(model_name):
    torch.manual_seed(19260817)
    np.random.seed(19260817)

    if model_name == 'fnn':
        fnn_hyperparams = {
            'learning_rate': [1e-3, 5e-4, 1e-4],
            'batch_size': [16, 32, 64], 
            'num_classes': [1],
            'h_size': [32, 64],
            'num_epochs': [100]
        }

        best_fnn_params, fnn_loss, fnn_acc = fnn_tuning(fnn_hyperparams)
        print("Best FNN Params:", best_fnn_params)
        print("FNN Validation Loss:", fnn_loss)
        print("FNN Validation Accuracy:", fnn_acc)

    elif model_name == 'cnn':
        cnn_hyperparams = {
            'num_filters': [100], #[50, 100, 150]
            'kernel_sizes': [(3, 4, 5)],
            'dropout': [0], #[0, 0.1, 0.2]
            'learning_rate': [1e-3, 5e-4],
            'batch_size': [32, 64],
            'num_classes': [1],
            'num_epochs': [40]
        }

        best_cnn_params, cnn_loss, cnn_acc = cnn_tuning(cnn_hyperparams)
        print("Best CNN Params:", best_cnn_params)
        print("CNN Validation Loss:", cnn_loss)
        print("CNN Validation Accuracy:", cnn_acc)

    elif model_name == 'transformer':
        transformer_hyperparams = {
            'num_heads': [4], #[4,8]
            'h_size': [256], #[128, 256]
            'num_layers': [2], #[1,2]
            'dropout': [0.1], #[0.1, 0.2]
            'learning_rate': [5e-5, 1e-4],
            'batch_size': [16], #[8, 16, 32]
            'num_classes': [1],
            'num_epochs': [40]
        }

        best_transformer_params, transformer_loss, transformer_acc = transformer_tuning(transformer_hyperparams)
        print("Best Transformer Params:", best_transformer_params)
        print("Transformer Validation Loss:", transformer_loss)
        print("Transformer Validation Accuracy:", transformer_acc)


    elif model_name == 'bert':
        bert_hyperparams = {
            'learning_rate': [2e-5, 3e-5],
            'batch_size': [16, 32],
            'num_classes': [1],
            'dropout': [0.1, 0.2],
            'num_epochs': [10]
        }

        best_bert_params, bert_loss, bert_acc = bert_tuning(bert_hyperparams)
        print("Best BERT Params:", best_bert_params)
        print("BERT Validation Loss:", bert_loss)
        print("BERT Validation Accuracy:", bert_acc)

    else:
        print("Invalid model name! Please use one of the following: fnn, cnn, transformer, bert")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python hyper_tuning.py <model_name>")
    else:
        model_name = sys.argv[1].lower()
        main(model_name)
