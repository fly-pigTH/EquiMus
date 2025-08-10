import torch
import torch.nn as nn
import torch.optim as optim

# 生成一些简单的线性数据
# y = 2x + 1 + noise
torch.manual_seed(42)  # 设置随机种子以保证结果可复现
x = torch.linspace(-10, 10, 100).reshape(-1, 1)  # 输入特征
y = 2 * x + 1 + torch.randn_like(x) * 2  # 添加噪声的目标值

# 定义一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 单变量线性回归

    def forward(self, x):
        return self.linear(x)

# 初始化模型、损失函数和优化器
model = LinearRegressionModel()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 使用 Adam 优化器，学习率设为 0.01

# 训练模型
epochs = 100000  # 训练轮数
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清零梯度

    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)  # 计算损失

    # 反向传播和优化
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 打印训练进度
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 打印训练后的模型参数
print("训练后的模型参数：")
print(f"Weight: {model.linear.weight.item():.4f}, Bias: {model.linear.bias.item():.4f}")