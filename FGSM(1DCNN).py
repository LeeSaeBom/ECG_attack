import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


# Define the CNNModel class
class CNNModel(torch.nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(11392, num_classes+1)  # Update this value based on actual output size

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Define the ECGDataset class
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


# Load the best model weights
best_model_weights_path = "./best_model_code/best_model_layer_1.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(input_channels=1, num_classes=100)  # Update num_classes based on your dataset
model.load_state_dict(torch.load(best_model_weights_path))
model.to(device)
model.eval()

# Load the dataset
dataset = ECGDataset("./one")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# FGSM parameters
epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]

for epsilon in epsilon_values:
    print(f"FGSM Attack with Epsilon = {epsilon:.2f}")

    f1_scores = []
    accuracies = []

    for data, labels in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        data.requires_grad = True

        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        attack_data = data + epsilon * data.grad.sign()
        attack_data = torch.clamp(attack_data, 0, 1)  # Ensure data stays in valid range [0, 1]

        # Evaluate the model with the attack data
        with torch.no_grad():
            attack_outputs = model(attack_data)
            attack_predictions = torch.argmax(attack_outputs, dim=1)
            f1 = f1_score(labels.cpu(), attack_predictions.cpu(), average='weighted')
            accuracy = accuracy_score(labels.cpu(), attack_predictions.cpu())
            f1_scores.append(f1)
            accuracies.append(accuracy)

    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracies)

    print(f"   Average F1 Score: {avg_f1:.4f}")
    print(f"   Average Accuracy: {avg_accuracy:.4f}")
    print()
