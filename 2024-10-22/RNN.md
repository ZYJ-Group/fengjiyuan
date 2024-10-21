**RNN**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. 数据准备
# 假设我们有一些简单的随机数据作为输入 (X) 和标签 (y)
# 100个样本，每个样本是长度为5的序列，每个时间步有10个特征
X = np.random.rand(100, 5, 10).astype(np.float32)  # (100, 5, 10)
y = np.random.randint(0, 2, size=(100, 1)).astype(np.float32)  # 二分类标签

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# 使用 DataLoader 来分批次加载数据
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2. 定义循环神经网络 (RNN)
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # 使用batch_first
        self.fc = nn.Linear(hidden_size, output_size)  # 输出层

    def forward(self, x):
        # x 的形状是 (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)  # RNN的输出
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = torch.sigmoid(self.fc(out))  # 通过全连接层并应用sigmoid激活
        return out

# 3. 初始化模型、损失函数和优化器
input_size = 10  # 输入特征的维度
hidden_size = 16  # RNN 隐藏层的神经元数量
output_size = 1  # 输出层的神经元数量（二分类）
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
epochs = 20  # 训练轮数
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()  # 清除上一次迭代中的梯度
        outputs = model(batch_X)  # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 测试模型（简单测试）
with torch.no_grad():
    test_input = torch.tensor(np.random.rand(10, 5, 10).astype(np.float32))  # 随机生成测试数据
    test_output = model(test_input)
    print("Test Output:", test_output)
```

`np.random.randint(low, high=None, size=None, dtype=int)`

**low**：生成随机数的下界（包含在内）。

**high**：生成随机数的上界（不包含在内）。如果只给出了 `low`，则生成 `[0, low)` 之间的随机数。

**size**：输出数组的形状。如果不指定，返回一个标量（单个数值）。可以是一个整数（表示一维数组的大小），也可以是一个元组（表示多维数组的形状）。

**dtype**：返回的数据类型，默认为 `int`。

```python
np.random.randint(0, 2, size=(100, 1)).astype(np.float32)
```

生成一个形状为 `(100, 1)` 的随机整数数组，范围在 `[0, 2)` 之间：

`np.random.randint(5, 15)`

生成一个 `[5, 15)` 之间的随机整数





`np.random.rand(100, 5, 10)` 是 NumPy 中用于生成随机浮点数数组的函数，它会生成一个形状为 `(100, 5, 10)` 的三维数组，数组中的元素是从均匀分布 `[0, 1)` 之间的随机数。





```python
def __init__(self, input_size, hidden_size, output_size):
```

### 什么时候可以不用参数的 `__init__`？

如果你的应用场景非常固定，例如你只处理特定的输入数据，且不需要改变网络结构，那么可以省略参数，让网络的结构在类内部直接固定下来。但是大多数深度学习任务都需要一定的灵活性，因此推荐使用带参数的 `__init__` 方法。



```python
out, _ = self.rnn(x)  # RNN的输出
```

这里调用了 `self.rnn(x)`，它将输入 `x` 通过 RNN 层。

**`out`** 是 RNN 的输出，通常是一个三维张量，形状为 `(batch_size, seq_len, hidden_size)`，每个时间步的输出都是 `hidden_size` 维度的向量。

`self.rnn` 返回两个值：输出序列和最后一个隐藏状态（我们不需要隐藏状态，所以用 `_` 来忽略它）。



```python
out = out[:, -1, :]  # 取最后一个时间步的输出
```

这一步是从 `out` 中提取序列最后一个时间步的输出。`out[:, -1, :]` 表示：

- **`:`** 表示保留所有的样本（即保留批次中的所有数据）。
- **`-1`** 表示选择最后一个时间步的输出（因为我们假设模型的决策主要基于最后一个时间步的隐藏状态）。
- **`:`** 表示保留最后一个时间步的所有特征（即隐藏层的所有神经元的输出）。