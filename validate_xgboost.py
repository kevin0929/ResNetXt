import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class FCNModel(nn.Module):
    def __init__(self, num_classes=6):
        super(FCNModel, self).__init__()

        # convolution layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        # batch normalization
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # global pooling layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # fully connected layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)  # ReLU activation

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = nn.ReLU()(x)

        x = self.global_pool(x)
        x = x.squeeze(dim=2)  # Remove the last dimension

        return x


def load_data(valid_data_path: str) -> TensorDataset:
    """_summary_

    Args:
        train_data_path (str): train data for npz format
        test_data_path (str): test data for npz format

    Returns:
        TensorDataset: for GPU training, transform to tensor
    """

    valid_data = np.load(valid_data_path)

    x_data = valid_data["x_valid"]
    y_data = valid_data["y_valid"]

    x_data = x_data.reshape(x_data.shape[0], 1, -1)

    # transform into pytorch tensor data
    x_data = torch.tensor(x_data).float()
    y_data = torch.tensor(y_data).long()

    # make TensorDataset
    valid_dataset = TensorDataset(x_data, y_data)

    return valid_dataset


def validate(model, device, valid_loader):
    model.eval()
    feature_list = []
    label_list = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            feature_list.append(outputs)
            label_list.append(labels)

    # Flatten the list of arrays into a single array
    features = np.concatenate(feature_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    return features, labels


if __name__ == "__main__":
    # convert data info DataLoader
    valid_data_path = "data/test_dataset7_cow560.npz"

    valid_dataset = load_data(valid_data_path)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    # build model and set config hyper parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNModel().to(device)

    model_path = "weights/model-ep150-val_loss1.193-val_acc0.851.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x_data, y_data = validate(
        model=model,
        device=device,
        valid_loader=valid_loader,
    )

    # Reshape x_data if necessary
    x_data = x_data.reshape(x_data.shape[0], -1)

    xgb_classifier = XGBClassifier(
        learning_rate=0.01,
        max_depth=6,
        n_estimators=200,
        objective="multi:softmax",
        num_class=6,
        random_state=42,
    )

    # Load pre-trained XGBoost model
    xgb_classifier.load_model("weights/xgboost_model_FCN.pt")

    # Predict and calculate accuracy
    y_pred = xgb_classifier.predict(x_data)
    print(accuracy_score(y_data, y_pred))
