import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split

class ECGDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = sorted(glob.glob(os.path.join(folder_path, "*", "*.csv")))
        self.data = []
        self.labels = []

        for file_path in self.file_list:
            # Load data
            data = pd.read_csv(file_path)

            # Data preprocessing
            data = data.iloc[:, 2:].values.astype(np.float32)

            # Append data and label
            self.data.append(data)
            label = os.path.basename(os.path.dirname(file_path))
            self.labels.append(int(label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        # Convert to tensor
        data_tensor = torch.from_numpy(data)
        label_tensor = torch.tensor(label)

        return data_tensor, label_tensor


# Create dataset instance
dataset = ECGDataset('./one')

# Calculate data split ratios
total_data = len(dataset)
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

train_size = int(total_data * train_ratio)
val_size = int(total_data * val_ratio)
test_size = total_data - train_size - val_size

# Split the dataset
train_dataset, testval_dataset = train_test_split(dataset, test_size=1 - train_ratio)
val_dataset, test_dataset = train_test_split(testval_dataset, test_size=test_ratio / (test_ratio + val_ratio))

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define BLSTM model and CUDA settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("장치 확인 : ", device)

input_size = dataset[0][0].shape[1]
hidden_size = 64
num_layers = 2
num_classes = 100

class BLSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size*2, num_classes + 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        out, (hidden, cells) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1])
        return out


model = BLSTMModel(input_size, hidden_size, batch_size, num_classes).to(device)

# Training settings
num_epochs = 500
learning_rate = 0.0001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []

    for data, labels in tqdm(data_loader):
        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    accuracy = total_correct / total_samples
    f1=f1_score(y_true,y_pred,average='macro')
    return accuracy, total_loss, f1


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, labels in tqdm(data_loader):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = total_correct / total_samples
    loss = total_loss / len(data_loader)
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, loss, f1


if __name__ == '__main__':
    best_val_acc=0.0
    train_losses = []
    train_accuracies = []
    train_f1_scores = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        train_acc, train_loss, train_f1 = train(model, train_loader, optimizer, criterion)
        val_acc, val_loss, val_f1 = evaluate(model, val_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1_scores.append(train_f1)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        print()
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1 Score : {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            torch.save(model.state_dict(), f'best_model_layer_{num_layers}.pt')
    test_acc, test_loss, test_f1 = evaluate(model, test_loader)
    print(f"Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
    print(f"num_layers: {num_layers}, epoch: {epoch+1}")
    # Save training metrics to a text file
    with open(f'train_metrics_layer_{num_layers}.txt', 'w') as file:
        file.write('Epoch\tLoss\tAccuracy\tF1 Score\n')
        for epoch, loss, acc, f1 in zip(range(1, num_epochs + 1), train_losses, train_accuracies, train_f1_scores):
            file.write(f'{epoch}\t{loss:.4f}\t{acc:.4f}\t{f1:.4f}\n')

    # Save validation metrics to a text file
    with open(f'val_metrics_layer_{num_layers}.txt', 'w') as file:
        file.write('Epoch\tLoss\tAccuracy\tF1 Score\n')
        for epoch, loss, acc, f1 in zip(range(1, num_epochs + 1), val_losses, val_accuracies, val_f1_scores):
            file.write(f'{epoch}\t{loss:.4f}\t{acc:.4f}\t{f1:.4f}\n')