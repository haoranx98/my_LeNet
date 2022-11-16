import torch
from torch import nn
import sys
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

mnist_train = torchvision.datasets.MNIST(root='~/Datasets/MNIST', train=True, download=True,
                                         transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='~/Datasets/MNIST', train=False, download=True,
                                        transform=transforms.ToTensor())


def load_data_fashion_mnist(mnist_train, mnist_test, batch_size):
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(mnist_train, mnist_test, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    # def __new__(cls, *args, **kwargs):
    #     return super(LeNet).__init__()

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()  # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    num_list = [i + 1 for i in range(num_epochs)]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        loss_list.append(train_l_sum / batch_count)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)
    return num_list, loss_list, train_acc_list, test_acc_list


# net = LeNet()

if __name__ == '__main__':
    lr_list = [0.1, 0.05, 0.02, 0.01, 0.001]
    epoch_list = [10, 50, 100, 200, 500, 1000]
    arg_list = []

    for i in range(len(lr_list)):
        for j in range(len(epoch_list)):
            arg_list.append((lr_list[i], epoch_list[j]))

    for arg in arg_list:
        net = LeNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=arg[0])
        num_list, loss_list, train_acc_list, test_acc_list = train(net, train_iter, test_iter, batch_size,
                                                                   optimizer, device, arg[1])
        plt.plot(num_list, loss_list, color='r', label='loss')
        plt.plot(num_list, train_acc_list, color='g', label='train')
        plt.plot(num_list, test_acc_list, color='b', label='test')

        name = str(arg[0]) + '_' + str(arg[1])
        plt.title(name)
        plt.xlabel = 'epoch'
        plt.legend(loc="best")  # 图例
        file_name = name + '.jpg'
        plt.savefig(file_name)
        plt.show()
