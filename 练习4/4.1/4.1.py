import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# =========================
# 1. 数据预处理
# =========================

transform = transforms.Compose([
    transforms.ToTensor(),

    # 标准化
    transforms.Normalize((0.5,), (0.5,))
])

# 自动下载 MNIST
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=64,
                         shuffle=False)

# =========================
# 2. 构建三层全连接网络
# =========================

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()

        # He初始化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):

        x = x.view(-1, 28*28)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = NeuralNet()

# =========================
# 3. 损失函数与优化器
# =========================

criterion = nn.CrossEntropyLoss()

# Adam优化器
#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.01)
# =========================
# 4. 开始训练
# =========================

epochs = 5

train_losses = []
train_accs = []

for epoch in range(epochs):

    correct = 0
    total = 0
    running_loss = 0

    model.train()

    for images, labels in train_loader:

        # 前向传播
        outputs = model(images)

        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = correct / total

    train_losses.append(epoch_loss)

    train_accs.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {epoch_loss:.4f} "
          f"Accuracy: {epoch_acc:.4f}")

# =========================
# 5. 可视化
# =========================

plt.figure(figsize=(10,4))

# Loss曲线
plt.subplot(1,2,1)

plt.plot(train_losses)

plt.title("Loss Curve")

plt.xlabel("Epoch")

plt.ylabel("Loss")

# Accuracy曲线
plt.subplot(1,2,2)

plt.plot(train_accs)

plt.title("Accuracy Curve")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.show()