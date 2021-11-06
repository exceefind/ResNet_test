import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import math
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def Conv_1x1(in_planes: int, planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, planes, 1, stride, bias=False)


def Conv_3x3(in_planes: int, planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, planes, 3, stride, padding=1, bias=False)


class ResNet_Block(nn.Module):
    Expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample=None):
        super(ResNet_Block, self).__init__()
        self.left = nn.Sequential(
            Conv_1x1(in_planes, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            Conv_3x3(planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            Conv_1x1(planes, planes * self.Expansion),
            nn.BatchNorm2d(planes * self.Expansion),
        )
        self.right = downsample

    def forward(self, x):
        Residual = x
        out = self.left(x)
        if self.right is not None:
            Residual = self.right(x)
        # print(x.size())
        # print(out.size())
        out += Residual

        return nn.ReLU(inplace=True)(out)


class ResNet(nn.Module):
    def __init__(self, num_class=100):
        super(ResNet, self).__init__()
        layers = [3, 4, 6, 3]
        # self.Conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Bn1 = nn.BatchNorm2d(64)
        self.Relu = nn.ReLU(inplace=True)
        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_1 = self._make_layer(64, 64, layers[0])
        self.layer_2 = self._make_layer(256, 128, layers[1], 2)
        self.layer_3 = self._make_layer(512, 256, layers[2], 2)
        self.layer_4 = self._make_layer(1024, 512, layers[3], 2)
        # self.avgPool=nn.AvgPool2d(7,1)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_class)
        # self.drop = nn.Dropout(p=0.5)
        # for p in self.parameters():
        #     p.requires_grad = False

    # def Weight_init(self):
    #     for m in self.module:
    #         if isinstance(m,nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data,a=0,nonlinearity='relu')
    #         elif isinstance(m,nn.BatchNorm2d) :
    #             nn.init.normal_(m.weight.data)

    def _make_layer(self, in_planes: int, planes: int, num_block: int, stride: int = 1):
        downsample = None
        if stride != 1 or in_planes != planes * ResNet_Block.Expansion:
            downsample = nn.Sequential(
                Conv_1x1(in_planes, planes * ResNet_Block.Expansion, stride),
                nn.BatchNorm2d(planes * ResNet_Block.Expansion)
            )
        layer = []
        layer.append(ResNet_Block(in_planes, planes, stride, downsample))
        for i in range(1, num_block):
            layer.append(ResNet_Block(planes * ResNet_Block.Expansion, planes))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Bn1(x)
        x = self.Relu(x)
        # x=self.Maxpool(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        #        x=self.drop(x)
        return self.fc(x)


# def Update_lr(optimizer,lr,epoch):
#     for param_group in optimizer.param_groups:
#         param_group['lr']=

def compute_mean_std(cifar100_dataset):
    # print(cifar100_dataset.data.shape)
    data_r = np.dstack([cifar100_dataset.data[i][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset.data[i][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset.data[i][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)
    return mean, std


def draw_train(loss, train_acc, dev_acc, epoch_num):
    epochs = range(epoch_num)
    plt.plot(epochs, train_acc, color='blue')
    plt.plot(epochs, dev_acc, color='red')
    plt.show()


if __name__ == '__main__':
    # 模型超参数设置：
    Num_Epoch = 200
    Batch_size = 64
    da_num = 1
    Lr = 8e-2
    # Lr=1e-4
    #     指数移动平均, bm为移动参数β
    loss_moving = 0
    bm = 0.99
    ACC = 0
    MILESTONE = [60, 120, 160]
    # MILESTONE = [20, 40, 60, 80]
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    mean, std = compute_mean_std(torchvision.datasets.CIFAR100(root='../data/',
                                                               transform=transforms.ToTensor(),
                                                               download=False))
    # 数据加载：
    # 设置transform  采取随机剪裁、以0.5概率进行水平旋转，增强模型的泛化能力
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(np.array(mean) / 255, np.array(std) / 255)
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    augment_1 = transforms.Compose([
        # transforms.RandomCrop(32, padding=2),
        transforms.ColorJitter(hue=0.5),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array(mean) / 255, np.array(std) / 255)
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载数据集
    # CIFAR-100 数据集下载
    train_dataset = torchvision.datasets.CIFAR100(root='../data/',
                                                   train=True,
                                                   transform=transform,
                                                   download=False)

    train_dataset2 = torchvision.datasets.CIFAR100(root='../data/',
                                                   train=True,
                                                   transform=augment_1,
                                                   download=False)

    # train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])

    test_dataset = torchvision.datasets.CIFAR100(root='../data/',
                                                 train=False,
                                                 transform=test_transform)
    # 数据载入
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=Batch_size,
                                               shuffle=True, num_workers=8)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=Batch_size,
                                              shuffle=False, num_workers=8)

    #    print(len(train_loader))
    #    train_dataset_aug = torchvision.datasets.CIFAR100(root='../data/',
    #                                                 train=True,
    #                                                 transform=augment_1 ,
    #                                                 download=False)
    #    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset_aug,
    #                                               batch_size=Batch_size,
    #                                               shuffle=True)

    ResNet = ResNet().to(device)

    model_dict = ResNet.state_dict()
    print(list(model_dict.keys()))

    param_tr = []
    for m in ResNet.modules():
        if isinstance(m,nn.BatchNorm2d):
             # m.parameters()表示参数对象
            # print(m.parameters())
            param_tr += m.parameters()
    param_rest = filter(lambda x:id(x) not in list(map(id,param_tr)),ResNet.parameters())

    #     定义优化器与学习率衰减
    #     optimizer=torch.optim.Adam(ResNet.parameters(),Lr,weight_decay=5e-4)
    # optimizer = torch.optim.SGD(ResNet.parameters(), Lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam([{'params':param_tr,'lr':Lr,},
                                  {'params':param_rest,'lr':0}
                                  ])
    #     Schuler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min=0)
    Schuler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONE, gamma=0.2)
    loss_func = nn.CrossEntropyLoss()
    # ResNet.load_state_dict(torch.load('../Resnet50_100_2021_09_14_12:10:45.ckpt'))

    # for p in optimizer.param_groups:
    #     outputs = ''
    #     for k, v in p.items():
    #         if k == 'params':
    #             for i in v:
    #                 outputs += (k + ': ' + str(i.shape).ljust(30) + ' ')
    #             outputs += '\n'
    #         else:
    #             outputs += (k + ': ' + str(v).ljust(10) + ' ')
    #     print(outputs)

    train_accs = []
    Loss = []
    dev_accs = []
    print("Train Start...")
    for epoch in tqdm(range(Num_Epoch)):
        sum_loss = 0
        correct = 0
        total = 0
        train_acc = 0
        ResNet.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            out = ResNet(images)
            _, predict = torch.max(out.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)

            loss = loss_func(out, labels)
            loss_moving = (1 - bm) * loss.item() + bm * loss_moving
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc = correct / total * 100
            if (i + 1) % 100 == 0:
                print("Epoch {} of {}: Batch {} of {}  loss: {:4f}  Training Acc: {:3f}".format(epoch + 1, Num_Epoch,
                                                                                                i + 1,
                                                                                                math.ceil(
                                                                                                    50000 * da_num / Batch_size),
                                                                                                loss_moving, train_acc))
        Loss.append(loss_moving)
        train_accs.append(train_acc)
        Schuler.step()

        if epoch%20 == 0:
            for p in optimizer.param_groups:
                outputs = ''
                for k, v in p.items():
                    if k == 'params':
                        for i in v:
                            outputs += (k + ': ' + str(i.shape).ljust(30) + ' ')
                        outputs += '\n'
                    else:
                        outputs += (k + ': ' + str(v).ljust(10) + ' ')
                print(outputs)

        ResNet.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            dev_acc = 0
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                predict = ResNet(images)
                _, pre = torch.max(predict, 1)

                correct += (pre == labels).sum().item()
                total += labels.size(0)

            dev_acc = correct / total * 100
            print("Dev Acc : {:4f}".format(dev_acc))
            dev_accs.append(dev_acc)

        # if (epoch+1)%50==0:
        #     torch.save(ResNet.state_dict(),
        #                '../Resnet50_ACC=_' + str(ACC)+ '.ckpt')

        print("Learing Rate of Epoch {}: {}".format(epoch + 1, optimizer.state_dict()['param_groups'][0]['lr']))

    print("Train finish!!!")
    draw_train(Loss, train_accs, dev_accs, Num_Epoch)

    torch.save(ResNet.state_dict(),
               'Resnet50_ACC=_' + str(ACC) + '_' + time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime()) + '.ckpt')

    # torch.save({'state_dict': ResNet.state_dict()}, '../Resnet50_100e.pth.tar')
    # ResNet.load_state_dict(torch.load('../Resnet50.ckpt'))

    # ResNet.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for i, (images, labels) in enumerate(test_loader):
    #        images = images.to(device)
    #        labels = labels.to(device)
    #        predict = ResNet(images)
    #
    #        _, predict = torch.max(predict.data, 1)
    #        correct += (predict == labels).sum().item()
    #        total += labels.size(0)
    #
    #     print("Accuracy of ResNet_50 model in CIFAR-100 : {:4f}".format(correct / total * 100))


