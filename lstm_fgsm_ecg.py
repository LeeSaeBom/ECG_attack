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

layer=5

class ECGDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = sorted(glob.glob(os.path.join(folder_path, "*", "*.csv")))
        self.data = []
        self.labels = []

        for file_path in self.file_list:
            data = pd.read_csv(file_path)
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


model_checkpoint_path = f"best_model_layer_{layer}.pt"


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes + 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1])
        return out

dataset = ECGDataset('../one')

model = LSTMModel(input_size=dataset[0][0].shape[1], hidden_size = 64, num_layers = layer, num_classes = 100)

checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

model.load_state_dict(checkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.eval()

dataset = ECGDataset('../one')
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

        perturbed_data = fgsm_attack(model, data, target, epsilon)

        outputs = model(perturbed_data)
        _, predicted = torch.max(outputs.data, 1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

        true_labels.extend(target.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')  # Update the average parameter to 'weighted'
    print(f"layer : {layer}")
    print(f"Epsilon: {epsilon}, Accuracy: {accuracy:.4f}%, F1-Score: {f1:.4f}")
    results_file.write(f"Epsilon: {epsilon}, Accuracy: {accuracy:.4f}%, F1-Score: {f1:.4f}\n")

results_file.close()
