#layer1
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
import time

start_time=time.time()

num_layers=1

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
            data = np.expand_dims(data, axis=0)  # Add batch dimension
            data = np.transpose(data, axes=(0, 2, 1))  # Transpose to (batch, channel, sequence_length)

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

# Define 1D CNN model and CUDA settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_channels = 1 # Set input channels to 1
num_classes = 100

#layer1
class CNNModel(torch.nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(11392, num_classes + 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#layer2
# class CNNModel(torch.nn.Module):
#     def __init__(self, input_channels, num_classes):
#         super(CNNModel, self).__init__()
#         self.conv1 = torch.nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
#         self.relu1 = torch.nn.ReLU()
#         self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.relu2 = torch.nn.ReLU()
#         self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.fc = torch.nn.Linear(11392, num_classes + 1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

#layer3
# class CNNModel(torch.nn.Module):
#     def __init__(self, input_channels, num_classes):
#         super(CNNModel, self).__init__()
#         self.conv1 = torch.nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
#         self.relu1 = torch.nn.ReLU()
#         self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.relu2 = torch.nn.ReLU()
#         self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.relu3 = torch.nn.ReLU()
#         self.pool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.fc = torch.nn.Linear(11264, num_classes + 1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.pool3(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# layer4
# class CNNModel(torch.nn.Module):
#     def __init__(self, input_channels, num_classes):
#         super(CNNModel, self).__init__()
#         self.conv1 = torch.nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
#         self.relu1 = torch.nn.ReLU()
#         self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.relu2 = torch.nn.ReLU()
#         self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.relu3 = torch.nn.ReLU()
#         self.pool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv4 = torch.nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.relu4 = torch.nn.ReLU()
#         self.pool4 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.fc = torch.nn.Linear(11264, num_classes + 1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.pool3(x)
#         x = self.conv4(x)
#         x = self.relu4(x)
#         x = self.pool4(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x



model = CNNModel(input_channels, num_classes).to(device)

# Training settings
num_epochs = 100
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
        data = data.squeeze(2).to(device)  # Remove the extra dimension
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
    f1 = f1_score(y_true, y_pred, average='macro')
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
            data = data.squeeze(2).to(device)  # Remove the extra dimension
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
            torch.save(model.state_dict(), f'best_model_layer_{num_layers}.pth')
    test_acc, test_loss, test_f1 = evaluate(model, test_loader)
    print(f"Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
    print(f"num_layers: {num_layers}, epoch: {epoch+1}")
    # Save training metrics to a text file
    with open(f'train_val/train_metrics_layer_{num_layers}.txt', 'w') as file:
        file.write('Epoch\tLoss\tAccuracy\tF1 Score\n')
        for epoch, loss, acc, f1 in zip(range(1, num_epochs + 1), train_losses, train_accuracies, train_f1_scores):
            file.write(f'{epoch}\t{loss:.4f}\t{acc:.4f}\t{f1:.4f}\n')

    # Save validation metrics to a text file
    with open(f'train_val/val_metrics_layer_{num_layers}.txt', 'w') as file:
        file.write('Epoch\tLoss\tAccuracy\tF1 Score\n')
        for epoch, loss, acc, f1 in zip(range(1, num_epochs + 1), val_losses, val_accuracies, val_f1_scores):
            file.write(f'{epoch}\t{loss:.4f}\t{acc:.4f}\t{f1:.4f}\n')

end_time=time.time()
total_time=end_time-start_time

hours=int(total_time//3600)
minutes=int((total_time%3600)//60)
seconds=int(total_time%60)

print(f'{hours} hours, {minutes} minutes, {seconds} seconds')
