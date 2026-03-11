import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import pandas as pd
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import numpy as np
import random
from torchvision.models import resnet34, ResNet34_Weights

TT = 8

# Fix all possible random sources
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)

# Check whether a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Normalization and inverse normalization
class Normalizer:
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data):
        self.min_val = np.min(data)
        self.max_val = np.max(data)

    def transform(self, data):
        return (data - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, data):
        return data * (self.max_val - self.min_val) + self.min_val

# Define the dataset class
class ImageSequenceDataset(Dataset):
    def __init__(self, image_paths, labels, frames_per_sequence=TT, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.frames_per_sequence = frames_per_sequence
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sequence_paths = self.image_paths[idx]
        sequence = [Image.open(img_path) for img_path in sequence_paths]
        if self.transform:
            sequence = [self.transform(img) for img in sequence]
        sequence = torch.stack(sequence)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return sequence, label

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4476, 0.4310, 0.3838], std=[0.0262, 0.0266, 0.0300]),
])

# Load the dataset
def load_image_paths_and_labels(image_dirs, label_files, frames_per_sequence=TT):
    sequence_paths = []
    sequence_labels = []

    for dir_path, label_path in zip(image_dirs, label_files):
        label_data = pd.read_excel(label_path, usecols="B", skiprows=1, header=None)  # Assume velocity labels are in column B
        images = sorted(os.listdir(dir_path), key=lambda x: int(x.split('_')[1].split('.')[0]))

        for i in range(len(images) - frames_per_sequence + 1):
            sequence = images[i:i + frames_per_sequence]
            sequence_full_paths = [os.path.join(dir_path, img) for img in sequence]
            sequence_paths.append(sequence_full_paths)

            sequence_label = label_data.iloc[i + frames_per_sequence - 1, 0].astype(float)
            sequence_labels.append(sequence_label)

    return sequence_paths, np.array(sequence_labels)

image_folders = [f'C:/Users/Chen Jingning/Desktop/project2024/week5/1-1800平均周期数据/{i}' for i in
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
label_files = [f'C:/Users/Chen Jingning/Desktop/project2024/week5/1-1800平均周期数据/{i}.xlsx' for i in
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
sequence_paths, sequence_labels = load_image_paths_and_labels(image_folders, label_files)

# Initialize the Normalizer and normalize the velocity data
normalizer = Normalizer()
normalizer.fit(sequence_labels)  # Compute normalization parameters based on all labels
sequence_labels_normalized = normalizer.transform(sequence_labels)

# Split the normalized data
X_train, X_val, y_train, y_val = train_test_split(
    sequence_paths,
    sequence_labels_normalized,
    test_size=0.2,
    random_state=42
)

train_dataset = ImageSequenceDataset(X_train, y_train, transform=transform)
val_dataset = ImageSequenceDataset(X_val, y_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define the ResNet34 feature extractor
class ResNet34FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet34FeatureExtractor, self).__init__()
        pretrained_resnet34 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(pretrained_resnet34.children())[:-2])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

# Modified CNN + LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, time_steps=TT, hidden_size=2000, num_layers=3, dropout_rate=0.5):
        super(CNNLSTM, self).__init__()
        self.cnn = ResNet34FeatureExtractor()  # Use the ResNet34 feature extractor
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            input_size=4608,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        c_out = torch.zeros(batch_size, time_steps, 4608).to(x.device)
        for t in range(time_steps):
            c_out[:, t, :] = self.dropout1(self.cnn(x[:, t, :, :, :]))

        mean_feature = torch.mean(c_out, dim=1, keepdim=True)
        c_out = c_out - mean_feature

        r_out, (h_n, h_c) = self.lstm(c_out)
        output = self.fc(self.dropout2(r_out[:, -1, :]))
        return output

# Get predictions and true labels on the validation set
def get_predictions_and_labels(model, data_loader, device, normalizer):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for data in data_loader:
            inputs, label = data[0].to(device), data[1].to(device)
            output = model(inputs)
            predictions.extend(output.view(-1).cpu().numpy())
            labels.extend(label.view(-1).cpu().numpy())

    # Convert to NumPy arrays and then apply inverse normalization
    predictions = np.array(predictions)
    labels = np.array(labels)
    predictions = normalizer.inverse_transform(predictions)
    labels = normalizer.inverse_transform(labels)
    return predictions, labels

# Adjust model parameters and training configuration
model = CNNLSTM(time_steps=TT, hidden_size=2000, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
criterion = nn.MSELoss()

# Initialize early stopping parameters
best_val_loss = float('inf')
patience = 4
trials = 0
num_epochs = 35

train_losses = []
val_losses = []
plots_folder_path = 'C:/Users/Chen Jingning/Desktop/project2024/plots'
if not os.path.exists(plots_folder_path):
    os.makedirs(plots_folder_path)

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    training_loss = running_loss / len(train_loader)
    train_losses.append(training_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    end_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {training_loss:.7f}, Val Loss: {val_loss:.7f}, '
        f'Learning Rate: {current_lr}, Epoch Duration: {end_time - start_time:.2f} seconds'
    )

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trials = 0
        torch.save(model.state_dict(), 'cnn_lstm_water_speed_model.pth')
    else:
        trials += 1
        print(f'Validation loss did not improve ({trials}/{patience})')

    if trials >= patience:
        print('Early stopping triggered')
        break

    if epoch + 1 in [28, 29, 30, 31, 32, 33, 34, 35]:
        model_save_path = f'cnn_lstm_water_speed_model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved for epoch {epoch + 1} at {model_save_path}')

        predictions, true_labels = get_predictions_and_labels(model, val_loader, device, normalizer)

        mse = mean_squared_error(true_labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_labels, predictions)
        r2 = r2_score(true_labels, predictions)

        print(f'Epoch {epoch + 1}: Validation MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}')

        loss_df = pd.DataFrame({
            'Epoch': list(range(1, len(train_losses) + 1)),
            'Train_Loss': train_losses,
            'Val_Loss': val_losses
        })

        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
            'Value': [mse, rmse, mae, r2]
        })

        results_df = pd.DataFrame({
            'True_Labels': true_labels,
            'Predictions': predictions
        })

        excel_save_path = f'C:/Users/Chen Jingning/Desktop/project2024/epoch_{epoch + 1}_results.xlsx'

        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            loss_df.to_excel(writer, sheet_name='Loss', index=False)
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            results_df.to_excel(writer, sheet_name='Predictions_vs_True', index=False)

        print(f'Epoch {epoch + 1} results saved to {excel_save_path}')

        plot_filename = f'predicted_vs_actual_epoch_{epoch + 1}.png'
        plot_full_path = os.path.join(plots_folder_path, plot_filename)

        plt.figure(figsize=(10, 6))
        plt.scatter(true_labels, predictions, alpha=0.6)
        plt.xlabel('True Water Speed (m/s)')
        plt.ylabel('Predicted Water Speed (m/s)')
        plt.title(f'Predicted vs True Water Speeds at Epoch {epoch + 1}')
        plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], color='red')
        plt.savefig(plot_full_path)
        plt.close()
        print(f'Plot saved to {plot_full_path}')

model.load_state_dict(torch.load('cnn_lstm_water_speed_model.pth'))
torch.cuda.empty_cache()