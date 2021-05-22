import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import models
import numpy as np
from matplotlib import pyplot as plt

def plot_loss(loss):
    x = range(1, 101, 1)
    y = loss
    # ---------------------------------------------
    ax = plt.gca()  # 得到图像的Axes对象
    ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.title('')
    plt.xlabel('x')
    plt.ylabel('loss')
    plt.plot(x, y)
    plt.show()

def start_to_train(dr):
    input_size = 28 * 28  # every picture in MNIST consists of 28 * 28 pixels
    hidden_size = 500  # the number of hidden neurons
    num_classes = 10  # MNIST has 10 classes
    num_epoches = 100  # 100 epoches
    batch_size = 100
    learning_rate = 0.0001

    # use cuda or not
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load data
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = models.MLP5(input_size, hidden_size, num_classes).to(device)
    model.dropout_rate = dr
    model.train()

    critertion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = []

    train_steps = len(train_loader)
    for epoch in range(num_epoches):
        for i, (img, labels) in enumerate(train_loader):
            img = img.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(img)
            outputs = outputs.to(device)
            running_loss = critertion(outputs, labels)

            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()

        total_loss.append(running_loss.item())
        print('loss:{:.4f}'.format(running_loss.item()))
    plot_loss(total_loss)
    return torch.save(model, 'FourLayersMLP.pth')
