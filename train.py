import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    return ResNet(BasicBlock, [2, 2, 2, 2])

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # PART 1：DATA AUG
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=15),
        # transforms.RandomResizedCrop(SIZE),
        transforms.ToTensor(),
        # mean and std of CIFAR 10's train set
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Channels = 3->1
        #transforms.Grayscale()
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Grayscale()
    ])

    train_set = CIFAR10(
        root='venv/CIFAR10/',
        train=True,
        download=False,
        transform=transforms_train)

    test_set = CIFAR10(
        root='venv/CIFAR10/',
        train=False,
        download=False,
        transform=transforms_test)

    # PART 2: Model PARAMS
    net = ResNet(BasicBlock, [2, 2, 2, 2])
    net.to(device)

    # PART 3: Hyper PARAMS
    BATCH_SIZE = 64
    EPOCH = 10
    Optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    CELoss = CrossEntropyLoss()

    """
    Train Iters
    """
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    loss_arr = np.array(0)
    acc_arr = np.array(0)
    for epoch in range(1, EPOCH):
        for step, (x, y) in enumerate(train_loader):
            output = net(x)
            loss = CELoss(output, y)
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

        # Test iters for each epoch
        correct = 0
        total = 0
        for batch in test_loader:
            x, y = batch
            output = net(x)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum()
        acc = 100 * correct / total
        print('EPOCH:{epoch}, ACC:{acc}%, LOSS:{loss}'.format(epoch=epoch, acc=acc, loss=loss))
        loss_arr = np.hstack([loss_arr, loss.detach().numpy()])
        acc_arr = np.hstack([acc_arr, acc.detach().numpy()])

    plt.figure()
    plt.plot(loss_arr)
    plt.figure()
    plt.plot(acc_arr)
    plt.show()