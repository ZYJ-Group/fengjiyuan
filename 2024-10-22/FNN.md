```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. 数据准备
# 假设我们有一些简单的随机数据作为输入 (X) 和标签 (y)
# 100个样本，每个样本有10个特征
X = np.random.rand(100, 10).astype(np.float32)
# 标签 (0 或 1)，二分类任务
y = np.random.randint(0, 2, size=(100, 1)).astype(np.float32)

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# 使用 DataLoader 来分批次加载数据
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2. 定义前馈神经网络 (FNN)
class SimpleFNN(nn.Module):
    def __init__(self):
        super(SimpleFNN, self).__init__()
        # 输入层 -> 隐藏层 1 -> 隐藏层 2 -> 输出层
        self.fc1 = nn.Linear(10, 32)  # 输入10个特征，输出32个神经元
        self.fc2 = nn.Linear(32, 16)  # 输入32个神经元，输出16个神经元
        self.fc3 = nn.Linear(16, 1)   # 输入16个神经元，输出1个神经元（二分类）

    def forward(self, x):
        x = torch.relu(self.fc1(x))   # 使用ReLU激活函数
        x = torch.relu(self.fc2(x))   # 使用ReLU激活函数
        x = torch.sigmoid(self.fc3(x)) # 使用Sigmoid激活函数，适合二分类
        return x

# 3. 初始化模型、损失函数和优化器
model = SimpleFNN()
criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
epochs = 20  # 训练轮数
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()            # 清除梯度
        outputs = model(batch_X)         # 前向传播
        loss = criterion(outputs, batch_y) # 计算损失
        loss.backward()                  # 反向传播
        optimizer.step()                 # 更新参数

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 测试模型（简单测试）
with torch.no_grad():
    test_input = torch.tensor(np.random.rand(10, 10).astype(np.float32))  # 随机生成测试数据
    test_output = model(test_input)
    print("Test Output:", test_output)
```

`np.random.rand()`：返回一个在 `[0, 1)` 区间内的随机浮点数。

`np.random.rand(3)`：返回一个形状为 `(3,)` 的一维数组，包含三个随机数。

`np.random.rand(2, 3)`：返回一个形状为 `(2, 3)` 的二维数组，包含两行三列的随机数。





`TensorDataset` 可以将多个 `torch.Tensor` 组合成一个数据集，每次返回的元素就是这些张量对应位置上的数据。常用于特征和标签的数据集场景。





`optimizer.zero_grad()` 的作用是 **清除上一次迭代中累积的梯度**。

在训练神经网络时，每次通过 `backward()` 进行反向传播时，PyTorch 会计算当前批次数据的梯度，并将它们累积（即相加）到已经存储在模型参数中的梯度上。这个累积行为在某些情况下是有用的，例如通过多个小批次来模拟大批次（梯度累积技术）。但是，在大多数标准的训练循环中，我们希望每次计算梯度都是基于当前批次的数据，独立于之前批次的梯度。因此，需要在每个训练步骤开始时，手动将梯度清零。





**`torch.no_grad()`**：在这个上下文中，所有的张量操作都不会被跟踪，也就是 `requires_grad=False`，不构建计算图，也不会消耗额外的内存进行梯度存储。

测试集不需要计算梯度。