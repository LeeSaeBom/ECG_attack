#layer=3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import glob
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score

layer=3

class ECGDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = sorted(glob.glob(os.path.join(folder_path, "*", "*.csv")))
        self.data = []
        self.labels = []

        for file_path in self.file_list:
            data = pd.read_csv(file_path)

            # 전처리()
            data = data.iloc[:, 2:].values.astype(np.float32)
            data = np.expand_dims(data, axis=0)
            data = np.transpose(data, axes=(0, 2, 1))

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


def fgsm_attack(model, data, target, epsilon):
    model.train()

    data.requires_grad = True

    outputs = model(data)
    loss = nn.CrossEntropyLoss()(outputs, target)
    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data
    perturbed_data = data + epsilon * torch.sign(data_grad)

    return perturbed_data


model_checkpoint_path = "./best_model_code/best_model_layer_3.pth"


class CNNModel(torch.nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(11264, num_classes + 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

dataset = ECGDataset('./one')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(input_channels=1, num_classes=100).to(device)
model = model.to(device)

checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

model.load_state_dict(checkpoint)



model.eval()

dataset = ECGDataset('./one')
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

epsilons = [0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035,0.04,0.045,0.05,0.1,0.15,0.2,0.25,0.3]

results_file = open(f"fgsm/fgsm_results_layer_{layer}.txt", "a")

for epsilon in epsilons:
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []

    for data, target in tqdm(data_loader):
        data, target = data.cuda(), target.cuda()
        data=data.squeeze(2)
        perturbed_data = fgsm_attack(model, data, target, epsilon)

        outputs = model(perturbed_data)
        _, predicted = torch.max(outputs.data, 1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

        true_labels.extend(target.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"Epsilon: {epsilon}, Accuracy: {accuracy:.4f}%, F1-Score: {f1:.4f}")
    results_file.write(f"Epsilon: {epsilon}, Accuracy: {accuracy:.4f}%, F1-Score: {f1:.4f}\n")

results_file.close()
