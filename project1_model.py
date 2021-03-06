# -*- coding: utf-8 -*-
"""project1_model.py

ResNet train scripts by st4253@nyu.edu, cz2397@nyu.edu, zz3810@nyu.edu for ECE-GY-7123 2022 Spring Mini-project 1

"""

import ssl
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
    """
    A residual block in the residual layer of ResNet.
    """

    def __init__(self, in_planes: int, planes: int, kernel_size: int, padding: int, stride: int = 1):
        """
        :param in_planes: Number of channels of input tensor
        :param planes: Number of channels of output tensor
        :param kernel_size: Size of kernel for every convolution layer in this block
        :param padding: Size of padding for every convolution layer in this block
        :param stride: Size of stride for conv1 and shortcut's conv layer (for conv2, stride=1)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet implementation class with some params adjustable.
    """

    def __init__(self, num_layers: int, num_blocks: list, kernel_size: list, avg_pool_size: int, num_classes: int):
        """
        :param num_layers: Number of residual Layers in ResNet
        :param num_blocks: Number of blocks in each residual layer
        :param kernel_size: List of kernel size for each residual layer
        :param avg_pool_size: Size of average pooling filter
        :param num_classes: Number of classes
        Note:
            num_layers = len(num_blocks) == len(kernel_size).
            Due to the first conv is stride=2 for i>=2 layers, feature map shrink as 1/2 per layer,\
            the feature map before AvgPool2D is (IMG_SIZE / 2 ** (num_layers - 1),so for the 32*32 image, \
            num_layers <= 6, as 1/2 ** 5 = 1/32,\
            and avg_pool_size should not exceed the size of final feature map.
        """
        super(ResNet, self).__init__()
        img_size = 32
        planes = 64
        block = BasicBlock
        self.in_planes = 64
        self.avg_pool_size = avg_pool_size
        self.kernel_size = kernel_size
        # Adjust conv padding size to maintain the feature map's size, based on W' = (W-F+2P)/2 + 1
        self.padding_size = [int((k - 1) / 2) for k in self.kernel_size]

        """
        initial convolution
        """
        self.conv1 = nn.Conv2d(3, planes, kernel_size=self.kernel_size[0],
                               stride=1, padding=self.padding_size[0], bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        """
        make sequential layers
        """
        # layer_i's planes equals to 64 * 2^(i-1)
        # make first layer (first block is stride=1)
        layers = [self._make_layer(block, planes, num_blocks[0],
                                   self.kernel_size[0], padding=self.padding_size[0], stride=1)]
        # loop to make num_layers - 1 layers
        for i in range(1, num_layers):
            planes = planes * 2
            layers.append(self._make_layer(block, planes, num_blocks[i],
                                           self.kernel_size[i], padding=self.padding_size[i], stride=2))
            self.in_planes = planes
        self.layers = nn.Sequential(*layers)

        """
        make final fully convolution layer
        """
        # final feature map's size = IMG_SIZE / 2 ** (num_layers - 1)
        map_size = int(img_size / 2 ** (num_layers - 1))
        # after AvgPool2D, map's size = (size - avg_pool_size)/avg_pool_size + 1  (for pooling, stride = kernel_size)
        map_size = int((map_size - avg_pool_size) / avg_pool_size + 1)
        # so the Linear input planes = Channels * [map_size ** 2]
        self.linear = nn.Linear(self.in_planes * (map_size ** 2), num_classes)

    def _make_layer(self, block, planes, num_blocks, kernel_size, padding, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, padding, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, self.avg_pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.device('cuda:0')
    print(torch.cuda.get_device_name(0))

    """
    Data Augmentation
    """
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    """
    Dataset
    """
    train_set = CIFAR10(
        root='CIFAR10/',
        train=True,
        download=True,
        transform=transforms_train)

    test_set = CIFAR10(
        root='CIFAR10/',
        train=False,
        download=True,
        transform=transforms_test)

    """
    Hyper params and model architecture
    """
    BATCH_SIZE = 256
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    net = ResNet(num_layers=3,
                 num_blocks=[3, 5, 2], kernel_size=[5, 3, 3],
                 avg_pool_size=4, num_classes=10)
    net.to(device)
    CELoss = CrossEntropyLoss()
    Optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=200)

    EPOCH = 100
    train_curve = []
    log_interval = 10
    MAX_EPOCH = EPOCH
    loss_mean = 0

    loss_arr = np.array(0)
    acc_arr = np.array(0)

    train_loss_history = []
    test_loss_history = []
    accuracy_history = []

    """
    Train iters
    """
    for epoch in range(EPOCH):
        train_loss = 0.0
        test_loss = 0.0
        for i, data in enumerate(train_loader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            Optimizer.zero_grad()
            predicted_output = net(images)
            fit = CELoss(predicted_output, labels)
            fit.backward()
            Optimizer.step()
            train_loss += fit.item()

        correct = 0
        total = 0
        for i, data in enumerate(test_loader):
            with torch.no_grad():
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                predicted_output = net(images)
                fit = CELoss(predicted_output, labels)
                test_loss += fit.item()

                _, predicted = torch.max(predicted_output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        accuracy_history.append(accuracy)
        print('Epoch %s, Train loss %s, Test loss %s, Test accuracy %s' % (epoch, train_loss, test_loss, accuracy))

        np.save(arr=np.array(accuracy_history), file="acc-final-{}.npy".format(accuracy))
        np.save(arr=np.array(train_loss_history), file="train-loss-final-{}.npy".format(accuracy))
        np.save(arr=np.array(test_loss_history), file="test-loss-final-{}.npy".format(accuracy))

        model_path = './project1_model-{}-{}.pt'.format(epoch, accuracy)
        torch.save(net.state_dict(), model_path)

    accuracy4plot = [acc / 100 for acc in accuracy_history]
    plt.plot(range(EPOCH), train_loss_history, '-', linewidth=3, label='Train error')
    plt.plot(range(EPOCH), test_loss_history, '-', linewidth=3, label='Test error')
    plt.plot(range(EPOCH), accuracy4plot, '-', linewidth=3, label='Accuracy')
    plt.xlabel('epoch')
    # plt.title('random horizontal flip -- random vertical flip')
    plt.grid(True)
    plt.legend()

    plt.savefig('final-version.png')
