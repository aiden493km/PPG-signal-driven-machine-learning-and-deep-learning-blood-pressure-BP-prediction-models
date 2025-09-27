from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# 自定义SE模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.se_ratio = se_ratio

        self.squeeze = nn.AdaptiveAvgPool1d(1)

        self.excitation = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // self.se_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // self.se_ratio, self.in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x).permute(0, 2, 1)
        out = self.excitation(out).permute(0, 2, 1)
        return x * out.expand_as(x)

# 自定义残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.se_block = SEBlock(out_channels, se_ratio=16)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se_block(out)
        out += identity
        out = self.relu(out)
        return out

# 自定义ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 定义ResNet-SE模型
def resnet_se(block, layers, num_classes=1):
    return ResNet(block, layers, num_classes)

# 读取Excel数据
data = pd.read_excel("F:\\SURF\\BP\\DATA.xlsx", sheet_name="总数据（男1女0）")

# 选择特征列和标签列
features_columns = ['VI', 'PT', 'ATD', 'DTD',
                    'Tsw10_mean', 'Tsw90_mean', 'Tdw50_mean', 'Tdw90_mean',
                    'Age', 'Height (cm)', 'Weight (kg)', 'BMI']
features = data[features_columns]
systolic_pressure = data['systolic pressure']
diastolic_pressure = data['diastolic pressure']

# 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
systolic_pressure_scaled = scaler.fit_transform(systolic_pressure.values.reshape(-1, 1))
diastolic_pressure_scaled = scaler.fit_transform(diastolic_pressure.values.reshape(-1, 1))

# 转换为PyTorch张量
features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1)
systolic_pressure_tensor = torch.tensor(systolic_pressure_scaled, dtype=torch.float32)
diastolic_pressure_tensor = torch.tensor(diastolic_pressure_scaled, dtype=torch.float32)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义损失函数
criterion = nn.MSELoss()

# 初始化KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储结果
all_pred_systolic, all_actual_systolic = [], []
all_pred_diastolic, all_actual_diastolic = [], []

# 5倍交叉验证
for fold, (train_idx, test_idx) in enumerate(kf.split(features_tensor)):
    print(f"Fold {fold+1}")

    # 生成训练和测试数据集
    X_train, X_test = features_tensor[train_idx], features_tensor[test_idx]
    y_train_systolic, y_test_systolic = systolic_pressure_tensor[train_idx], systolic_pressure_tensor[test_idx]
    y_train_diastolic, y_test_diastolic = diastolic_pressure_tensor[train_idx], diastolic_pressure_tensor[test_idx]

    train_dataset_systolic = TensorDataset(X_train, y_train_systolic)
    test_dataset_systolic = TensorDataset(X_test, y_test_systolic)
    train_dataset_diastolic = TensorDataset(X_train, y_train_diastolic)
    test_dataset_diastolic = TensorDataset(X_test, y_test_diastolic)

    train_loader_systolic = DataLoader(train_dataset_systolic, batch_size=64, shuffle=True)
    test_loader_systolic = DataLoader(test_dataset_systolic, batch_size=64, shuffle=False)
    train_loader_diastolic = DataLoader(train_dataset_diastolic, batch_size=64, shuffle=True)
    test_loader_diastolic = DataLoader(test_dataset_diastolic, batch_size=64, shuffle=False)

    # 初始化模型
    model_systolic = resnet_se(ResidualBlock, [2, 2, 2, 2]).to(device)
    model_diastolic = resnet_se(ResidualBlock, [2, 2, 2, 2]).to(device)

    # 定义优化器
    optimizer_systolic = torch.optim.Adam(model_systolic.parameters(), lr=0.01)
    optimizer_diastolic = torch.optim.Adam(model_diastolic.parameters(), lr=0.01)

    # 训练模型
    def train_model(model, train_loader, optimizer):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # 训练循环
    epochs = 50
    for epoch in range(epochs):
        train_model(model_systolic, train_loader_systolic, optimizer_systolic)
        train_model(model_diastolic, train_loader_diastolic, optimizer_diastolic)
        print(f"Epoch {epoch+1}/{epochs} completed for fold {fold+1}")


    # 模型评估
    def evaluate_model(model, test_loader):
        model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())
        if len(predictions) == 0 or len(actuals) == 0:
            raise ValueError("Test loader returned no data. Check your dataset or DataLoader.")
        return np.concatenate(predictions), np.concatenate(actuals)


    # 5倍交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_pred_systolic, all_actual_systolic = [], []
    all_pred_diastolic, all_actual_diastolic = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(features_scaled)):
        print(f"Fold {fold + 1}: Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        # 确保测试集不为空
        if len(test_idx) == 0:
            raise ValueError(f"Fold {fold + 1} has an empty test set, which may cause errors.")

        # 训练和测试数据集
        X_train_fold, X_test_fold = features_scaled[train_idx], features_scaled[test_idx]
        y_train_systolic_fold, y_test_systolic_fold = systolic_pressure_scaled[train_idx], systolic_pressure_scaled[
            test_idx]
        y_train_diastolic_fold, y_test_diastolic_fold = diastolic_pressure_scaled[train_idx], diastolic_pressure_scaled[
            test_idx]

        # 转换为PyTorch张量并创建数据集和数据加载器
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).unsqueeze(1)  # 增加一个维度以适应Conv1d
        X_test_tensor = torch.tensor(X_test_fold, dtype=torch.float32).unsqueeze(1)
        y_train_systolic_tensor = torch.tensor(y_train_systolic_fold, dtype=torch.float32)
        y_test_systolic_tensor = torch.tensor(y_test_systolic_fold, dtype=torch.float32)
        y_train_diastolic_tensor = torch.tensor(y_train_diastolic_fold, dtype=torch.float32)
        y_test_diastolic_tensor = torch.tensor(y_test_diastolic_fold, dtype=torch.float32)

        train_dataset_systolic = TensorDataset(X_train_tensor, y_train_systolic_tensor)
        test_dataset_systolic = TensorDataset(X_test_tensor, y_test_systolic_tensor)
        train_dataset_diastolic = TensorDataset(X_train_tensor, y_train_diastolic_tensor)
        test_dataset_diastolic = TensorDataset(X_test_tensor, y_test_diastolic_tensor)

        train_loader_systolic = DataLoader(train_dataset_systolic, batch_size=64, shuffle=True)
        test_loader_systolic = DataLoader(test_dataset_systolic, batch_size=64, shuffle=False)
        train_loader_diastolic = DataLoader(train_dataset_diastolic, batch_size=64, shuffle=True)
        test_loader_diastolic = DataLoader(test_dataset_diastolic, batch_size=64, shuffle=False)

        # 初始化模型
        model_systolic = resnet_se(ResidualBlock, [2, 2, 2, 2]).to(device)
        model_diastolic = resnet_se(ResidualBlock, [2, 2, 2, 2]).to(device)

        # 定义优化器
        optimizer_systolic = torch.optim.Adam(model_systolic.parameters(), lr=0.01)
        optimizer_diastolic = torch.optim.Adam(model_diastolic.parameters(), lr=0.01)

        # 训练模型
        for epoch in range(epochs):
            train_model(model_systolic, train_loader_systolic, optimizer_systolic)
            train_model(model_diastolic, train_loader_diastolic, optimizer_diastolic)

        # 评估模型
        pred_systolic, actual_systolic = evaluate_model(model_systolic, test_loader_systolic)
        pred_diastolic, actual_diastolic = evaluate_model(model_diastolic, test_loader_diastolic)

        # 存储结果
        all_pred_systolic.append(pred_systolic)
        all_actual_systolic.append(actual_systolic)
        all_pred_diastolic.append(pred_diastolic)
        all_actual_diastolic.append(actual_diastolic)

    # 合并预测结果
    all_pred_systolic = np.concatenate(all_pred_systolic)
    all_actual_systolic = np.concatenate(all_actual_systolic)
    all_pred_diastolic = np.concatenate(all_pred_diastolic)
    all_actual_diastolic = np.concatenate(all_actual_diastolic)

    # 反向标准化结果
    all_pred_systolic_unscaled = scaler.inverse_transform(all_pred_systolic)
    all_actual_systolic_unscaled = scaler.inverse_transform(all_actual_systolic)
    all_pred_diastolic_unscaled = scaler.inverse_transform(all_pred_diastolic)
    all_actual_diastolic_unscaled = scaler.inverse_transform(all_actual_diastolic)

    # 将结果保存为Excel文件
    output_df = pd.DataFrame({
        'Predicted Systolic': all_pred_systolic_unscaled.flatten(),
        'Actual Systolic': all_actual_systolic_unscaled.flatten(),
        'Predicted Diastolic': all_pred_diastolic_unscaled.flatten(),
        'Actual Diastolic': all_actual_diastolic_unscaled.flatten()
    })
    output_df.to_excel("D:\\SURF\\220BPsem5-1.xlsx", index=False)
    
print("Cross-validation completed and results saved.")
