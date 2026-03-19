import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from torchvision.models import resnet34, ResNet34_Weights

tt = 8

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
    def __init__(self, time_steps=tt, hidden_size=1000, num_layers=3, dropout_rate=0.5):
        super(CNNLSTM, self).__init__()
        self.cnn = ResNet34FeatureExtractor()  # Updated to use the ResNet34 feature extractor
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

# Define the Normalizer class for 0-1 normalization
class Normalizer:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, labels):
        self.min = np.min(labels)
        self.max = np.max(labels)

    def transform(self, labels):
        return (labels - self.min) / (self.max - self.min)

    def inverse_transform(self, normalized_labels):
        return normalized_labels * (self.max - self.min) + self.min

# Path to the model weight file
model_path = "C:/Users/Chen Jingning/Desktop/project2024/week5/1-1800平均周期/resnet34/4/30/4.pth"

# Instantiate the model and load the weights
model = CNNLSTM()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4476, 0.4310, 0.3838], std=[0.0262, 0.0266, 0.0300]),
])

# Load all label data to compute normalization parameters
label_files = [f'C:/Users/Chen Jingning/Desktop/project2024/week5/1-1800平均周期数据/{i}.xlsx' for i in range(1, 13)]
all_labels = []
for label_file in label_files:
    df = pd.read_excel(label_file, usecols=[1])  # Assume the labels are in the second column
    all_labels.extend(df.iloc[:, 0].values)
all_labels = np.array(all_labels)

# Initialize the Normalizer
normalizer = Normalizer()
normalizer.fit(all_labels)

# Define folder names and their corresponding Excel filenames
folders_and_excels = {
    '1': '1.xlsx',
    '2': '2.xlsx',
    '3': '3.xlsx',
    '4': '4.xlsx',
    '5': '5.xlsx',
    '6': '6.xlsx',
    '7': '7.xlsx',
    '8': '8.xlsx',
    '9': '9.xlsx',
    '10': '10.xlsx',
    '11': '11.xlsx',
    '12': '12.xlsx',
    '13': '13.xlsx',
    '14': '14.xlsx',
}

# Process each folder and its corresponding Excel file one by one
for folder_name, excel_name in folders_and_excels.items():
    # Path to the image folder
    image_folder = f'C:/Users/Chen Jingning/Desktop/project2024/1-12/{folder_name}'
    image_filenames = sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Split images into time sequences and preprocess them
    predicted_speeds_normalized = []
    sequence_length = tt  # Define the length of each sequence
    for i in range(len(image_filenames) - sequence_length + 1):
        imgs = [
            transform(Image.open(os.path.join(image_folder, fname)).convert('RGB'))
            for fname in image_filenames[i:i + sequence_length]
        ]
        imgs_tensor = torch.stack(imgs).unsqueeze(0).to(device)  # Move to the correct device
        with torch.no_grad():
            prediction = model(imgs_tensor).cpu().numpy().flatten()[0]
        predicted_speeds_normalized.append(prediction)

    # Apply inverse normalization to the predicted speed values
    predicted_speeds = normalizer.inverse_transform(np.array(predicted_speeds_normalized))

    # Read the true speed values
    true_speeds_path = f"C:/Users/Chen Jingning/Desktop/project2024/1-12 平均/4/{excel_name}"
    true_speeds_df = pd.read_excel(true_speeds_path, usecols=[1])  # Assume the true speeds are in the second column
    true_speeds = true_speeds_df.iloc[:, 0].values

    # Ensure that the number of true speeds matches the number of predicted speeds
    assert len(predicted_speeds) == len(true_speeds), (
        f"The number of predicted speeds does not match the number of true speeds for folder {folder_name}."
    )

    # Compute evaluation metrics
    mse = mean_squared_error(true_speeds, predicted_speeds)
    rmse = sqrt(mse)
    mae = mean_absolute_error(true_speeds, predicted_speeds)
    r_squared = r2_score(true_speeds, predicted_speeds)
    rms_true = np.sqrt(np.mean(np.square(true_speeds)))
    rmse_rms_ratio = rmse / rms_true

    # Save the results
    results_df = pd.DataFrame({
        'Sequence Number': np.arange(1, len(predicted_speeds) + 1),
        'Predicted Speed': predicted_speeds,
        'True Speed': true_speeds,
        'MSE': [mse] * len(predicted_speeds),
        'RMSE': [rmse] * len(predicted_speeds),
        'MAE': [mae] * len(predicted_speeds),
        'R²': [r_squared] * len(predicted_speeds),
        'RMSE/RMS': [rmse_rms_ratio] * len(predicted_speeds)
    })

    excel_filename_with_metrics = f'C:/Users/Chen Jingning/Desktop/project2024/week5/1-1800平均周期/resnet34/4/30/{folder_name}_metrics.xlsx'
    results_df.to_excel(excel_filename_with_metrics, index=False)

    print(f"Predicted speeds, true speeds, and metrics saved to {excel_filename_with_metrics} for folder {folder_name}")