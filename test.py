import torchvision
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import numpy as np


def start_to_test():

    # use cuda or not
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),
                                              download=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = torch.load('FiveLayersMLP.pth')
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for img, labels in test_loader:
            img = img.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('tes_acc = ', 100 * correct / total, '%')
        return 100 * correct / total
