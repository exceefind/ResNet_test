import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,shortcut=None):
        super (ResidualBlock, self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.right = shortcut

    def forward(self,x):
        residual = x
        out=self.left(x)
        if self.right:
            residual=self.right(x)
        out+=residual
        return nn.ReLU(inplace=True)(out)

# 分为4个layer，各层的residualBlock数为[3,4,6,3]
class ResNet(nn.Module):
    def __init__(self ,num_class=10):
        super(ResNet, self).__init__()
        self.pre=nn.Sequential(
            # 以7x7卷积核 步幅=2 填充=3 过滤 得到16*16*64
            nn.Conv2d(3, 64,  kernel_size=7,
                     stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, 1, 1)
        )

    #    定义4个layer，1个avgpool和1个fc
        self.layer1 = self.makeLayer(64, 64, 3)
        self.layer2 = self.makeLayer(64, 128, 4, 2)
        self.layer3 = self.makeLayer(128, 256, 6, 2)
        self.layer4 = self.makeLayer(256, 512, 3, 2)
        self.avgPool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512,num_class)

    def makeLayer(self,in_channel,out_channel,num_block,stride=1):
        shortcut=nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride),
            nn.BatchNorm2d(out_channel)
        )
        layer=[]
        layer.append(ResidualBlock(in_channel,out_channel,stride,shortcut))
        for i in range(num_block-1):
            layer.append(ResidualBlock(out_channel, out_channel))
        return nn.Sequential(*layer)

    def forward(self,x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgPool(x)
        x=x.view(x.size(0),-1)
        x = self.fc(x)
        return x

# 更新学习率
def update_LR(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义批数大小、代数、学习率
    BATCH_SIZE=64
    NUM_EPOCH=50
    Lr_base=0.01
    LR=Lr_base

    # 设置transform  采取随机剪裁、以0.5概率进行水平旋转，增强模型的泛化能力
    transform=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])

    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])

    # 加载数据集
    # CIFAR-10 数据集下载
    train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                                train=False,
                                                transform=test_transform)
    # 数据载入
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    model = ResNet().to(device)
    # 定义损失函数loss，以及优化器
    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)

    print("-------------------Train Start -------------------------")
    for epoch in range(NUM_EPOCH):
        sum_loss=0
        for i, (images, labels) in enumerate(train_loader):
            images=images.to(device)
            labels=labels.to(device)
            out = model(images)
            loss = loss_func(out,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print("Epoch {} of {}: Batch {} of {}  loss: {}".format(epoch+1, NUM_EPOCH, i+1, math.ceil(50000/BATCH_SIZE), loss.item()))

        # 保存训练过程的模型参数，分析准确率变化
        # if(epoch+1)%5 == 0:
        #     torch.save(model.state_dict(), './model_save/resnet34_'+str(epoch+1)+'.ckpt')

        # 动态调整学习率
        if (epoch+1)%30 == 0:
            LR=LR*0.1
            update_LR(optimizer,LR)

    print("-------------------Train finish ---------------------------")

    # model.load_state_dict(torch.load('./model_save/resnet34_50.ckpt'))
    # model.eval()
    # torch.save(model, 'resnet34.ckpt')

    model=torch.load('resnet34.ckpt')
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i,(images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            predict = model(images)
            _,predict=torch.max(predict.data,1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)

        print("Accuracy of ResNet model in CIFAR-10 : {:4f}".format(correct/total*100))


    acc=np.zeros(10)
    epo=np.linspace(5,50,10)
    for ep in epo:
        # model = ResNet().to(device)
        model.load_state_dict(torch.load('./model_save/resnet34_'+str(int(ep))+'.ckpt'))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                predict = model(images)
                _, predict = torch.max(predict.data, 1)
                correct += (predict == labels).sum().item()
                total += labels.size(0)
            print("Epoch "+str(int(ep))+": Accuracy of ResNet model in CIFAR-10 : {:4f}".format(correct / total * 100))
            index=int(ep/5)-1
            acc[index]=correct/total * 100
    acc=np.array(acc)

    plt.xlabel("Num of Epoch")
    plt.ylabel("Accuracy")
    plt.xlim(0, 50)
    plt.ylim(0, 100)
    plt.title('Changes of Top-1 Accuracy')
    plt.plot(epo,acc,linestyle="--",marker="o",markersize=8)
    for i in range(len(epo)):
        plt.text(epo[i], acc[i] + 2, '%.2f' % acc[i], ha='center', va='bottom', fontsize=9)
    plt.show()