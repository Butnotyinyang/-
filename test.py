import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from torch.optim import Adam, SGD

# 1 加载数据集
def get_dataloader(batch_size=128):
    # 训练集增强+标准归一化
    train_trans = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # 测试集同步归一化
    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_trans
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_trans
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

# 2 轻量化CNN网络 
class LightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 3 定义 SAM & DynamicSAM 
class SAM:
    def __init__(self, params, base_opt, rho=0.02):
        self.params = list(params)
        self.base_opt = base_opt
        self.rho = rho
        self.eps = 0
    def first_step(self):
        grads = []
        for p in self.params:
            if p.grad is not None:
                grads.append(p.grad.flatten())
        g_norm = torch.norm(torch.cat(grads))
        self.eps = self.rho / g_norm
        for p in self.params:
            if p.grad is not None:
                p.data += self.eps * p.grad
    def second_step(self):
        self.base_opt.step()
        for p in self.params:
            if p.grad is not None:
                p.data -= self.eps * p.grad
    def zero_grad(self):
        self.base_opt.zero_grad()

class DynamicSAM(SAM):
    def __init__(self, params, base_opt, total_epoch, init_rho=0.02):
        super().__init__(params, base_opt, init_rho)
        self.total_epoch = total_epoch
        self.init_rho = init_rho
    def update_rho(self, epoch):
        half = self.total_epoch * 0.5
        if epoch < half:
            self.rho = self.init_rho
        else:
            decay = (epoch - half) / half
            self.rho = self.init_rho * (1 - decay)

# 4 开始训练
if __name__ == "__main__":
    # 设置超参数
    EPOCHS = 24
    BATCH_SIZE = 64
    LR = 0.001       
    BASE_RHO = 0.02
    CRITERION = nn.CrossEntropyLoss()
    loss_none = nn.CrossEntropyLoss(reduction="none")
    # 切换优化器：sgd / adam / sam / dsam
    OPT_SELECT = "dsam"

    # 加载数据、网络
    train_loader, test_loader = get_dataloader(BATCH_SIZE)
    model = LightCNN()

    # 初始化优化器
    if OPT_SELECT == "sgd":
        base_opt = SGD(model.parameters(), lr=LR, weight_decay=0)
        opt = base_opt
    elif OPT_SELECT == "adam":
        opt = Adam(model.parameters(), lr=LR, weight_decay=0)
    elif OPT_SELECT == "sam":
        base_opt = Adam(model.parameters(), lr=LR, weight_decay=0)
        opt = SAM(model.parameters(), base_opt, rho=BASE_RHO)
    elif OPT_SELECT == "dsam":
        base_opt = Adam(model.parameters(), lr=LR, weight_decay=0)
        opt = DynamicSAM(model.parameters(), base_opt, total_epoch=EPOCHS)

    # 存储指标
    train_acc_record = []
    test_acc_record = []
    time_record = []
    log_file = open(f"{OPT_SELECT}_train_log.txt", "w", encoding="utf-8")

    # 开始训练循环
    for epoch in range(EPOCHS):
        start_time = time.time()
        if OPT_SELECT == "dsam":
            opt.update_rho(epoch)
        model.train()
        train_correct = 0
        train_loss_sum = 0

        for x, y in train_loader:
            opt.zero_grad()
            out = model(x)
            loss = CRITERION(out, y)
            loss.backward()

            if OPT_SELECT in ["sam", "dsam"]:
                opt.first_step()
                with torch.no_grad():
                    loss_batch = loss_none(out, y)
                _, top_idx = torch.topk(loss_batch, k=int(len(x)*0.5))
                loss_sub = CRITERION(model(x[top_idx]), y[top_idx])
                loss_sub.backward()
                opt.second_step()
                # 先统计数值，再释放张量
                train_loss_sum += loss.item()
                pred = torch.argmax(out, dim=1)
                train_correct += (pred == y).sum().item()
                del out, loss, loss_batch
            else:
                opt.step()
                train_loss_sum += loss.item()
                pred = torch.argmax(out, dim=1)
                train_correct += (pred == y).sum().item()
                del out, loss

        # 计算训练集精度
        train_acc = train_correct / len(train_loader.dataset)
        train_acc_record.append(train_acc)

        # 测试集评估
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                pred = torch.argmax(out, dim=1)
                test_correct += (pred == y).sum().item()
        test_acc = test_correct / len(test_loader.dataset)
        test_acc_record.append(test_acc)
        epoch_cost = time.time() - start_time
        time_record.append(epoch_cost)

        # 打印并写入日志
        log_info = f"Epoch {epoch+1:2d} | Train Acc:{train_acc:.4f} | Test Acc:{test_acc:.4f} | Time:{epoch_cost:.2f}s\n"
        print(log_info.strip())
        log_file.write(log_info)
    log_file.close()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 4))
    plt.plot(train_acc_record, label="Train Accuracy")
    plt.plot(test_acc_record, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{OPT_SELECT} Accuracy Curve")
    plt.legend()
    plt.savefig(f"./{OPT_SELECT}_acc_curve.png", bbox_inches="tight")
    plt.show()