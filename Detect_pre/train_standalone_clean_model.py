import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import os

# ==========================================
# 核心架构：适配 CIFAR-10 的 ResNet-18
# ==========================================
def get_cifar_resnet18():
    # 加载标准 ResNet18
    model = models.resnet18(weights=None)
    # 修改第一层卷积，适应 32x32 输入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 移除第一个 MaxPool，防止空间分辨率下降过快
    model.maxpool = nn.Identity()
    # 修改全连接层
    model.fc = nn.Linear(512, 10)
    return model

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  
    start_epoch = 0  

    # 1. 数据增强：这是达到 90%+ 的关键
    print("==> Preparing data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # 2. 模型、损失函数、优化器
    model = get_cifar_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    # 标准学术参数：LR=0.1, Momentum=0.9, Weight_Decay=5e-4
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # 学习率调度：在 50% 和 75% 的进度处衰减
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    # 3. 训练循环
    for epoch in range(100):
        print(f'\nEpoch: {epoch+1}/100')
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 4. 测试阶段
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        print(f'Test Acc: {acc:.2f}% | Best: {best_acc:.2f}%')
        
        # 保存最优模型
        if acc > best_acc:
            print('Saving model...')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_high_acc.pth')
            best_acc = acc
        
        scheduler.step()

if __name__ == '__main__':
    train()