import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# =========================
# 1. 数据预处理
# =========================
transform = transforms.Compose([
    #1. 随机水平翻转
    #transforms.RandomHorizontalFlip(),

    # 2. 随机旋转
    #transforms.RandomRotation(15),

    # 3. 随机裁剪
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# 自动下载 CIFAR-10
dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

# =========================
# 2. 定义CNN模型
# =========================
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 第1个卷积层
        self.conv1 = nn.Conv2d(
            in_channels=3,      #初始值为3
           out_channels=32,    #初始值为32
            #out_channels=64, 
            kernel_size=3,      #初始值为3
            #kernel_size=5,
            padding=1
        )

        # 第2个卷积层
        self.conv2 = nn.Conv2d(
            in_channels=32,     #初始值为32
            #in_channels=64,
            out_channels=64,    #初始值为64
            #out_channels=128,
            kernel_size=3,      #初始值为3
            #kernel_size=5,
            padding=1
        )

        # 池化层
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 全连接层
        self.fc = nn.Linear(64 * 8 * 8, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        # 卷积层1 + 激活 + 池化
        x = self.pool(
            self.relu(self.conv1(x))
        )

        # 卷积层2 + 激活 + 池化
        x = self.pool(
            self.relu(self.conv2(x))
        )

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接
        x = self.fc(x)

        return x

# =========================
# 3. 创建模型
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SimpleCNN().to(device)
print("Using device:", device)
# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# 4. 训练模型
# =========================
epochs = 20

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

for epoch in range(epochs):

    # ===== 训练 =====
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ===== 验证 =====
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print("-" * 50)

# =========================
# 5. 绘制曲线
# =========================

# Loss曲线
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')

plt.legend()

# Accuracy曲线
plt.subplot(1, 2, 2)

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')

plt.legend()

plt.show()