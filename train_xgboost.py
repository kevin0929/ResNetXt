import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.neighbors import KNeighborsClassifier
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


train_data = np.load("data/train_data.npz")
test_data = np.load("data/test_data.npz")

x_train = train_data["x_train"]
y_train = train_data["y_train"]
x_test = test_data["x_test"]
y_test = test_data["y_test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FCNModel().to(device)

# 如果您有保存的模型，請確保在加載時將其移至CPU
model_path = "weights/model-ep150-val_loss1.193-val_acc0.851.pth"
model.load_state_dict(torch.load(model_path))

x_train = x_train.reshape(x_train.shape[0], 1, -1)
x_test = x_test.reshape(x_test.shape[0], 1, -1)

# 將數據轉換為PyTorch張量並確保它們在CPU上
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).long()

x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).long()

# 設置DataLoader
batch_size = 32
train_loader = DataLoader(
    TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
)


# 定義特徵提取函數
def extract_features(loader, model):
    features = []
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            batch_features = model(x)
        features.append(batch_features.cpu().numpy())
    return np.concatenate(features, axis=0)


# 提取特徵
x_train = extract_features(train_loader, model)
x_test = extract_features(test_loader, model)

# use xgboost
xgb_classifier = XGBClassifier(
    learning_rate=0.01,
    max_depth=6,
    n_estimators=200,
    objective="multi:softmax",
    num_class=6,
    random_state=42,
)

# train
xgb_classifier.fit(x_train, y_train)

# test
y_pred = xgb_classifier.predict(x_test)

# save model
xgb_classifier.save_model("weights/xgboost_model_FCN.pt")
print(accuracy_score(y_test, y_pred))
