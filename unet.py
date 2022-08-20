import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor(),
])


# 定义数据库
class myDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('./data/horse'))

    def __getitem__(self, idx):
        pic_name = os.listdir('./data/horse')[idx]
        pic_1 = cv2.imread('./data/horse/' + pic_name)     #得到图片，并且对图片进行调整
        pic_1 = cv2.resize(pic_1, (160, 160))
        pic_2 = cv2.imread('./data/mask/' + pic_name, 0)
        pic_2 = cv2.resize(pic_2, (160, 160))
        pic_2 = torch.LongTensor(pic_2)
        if self.transform:
            pic_1 = self.transform(pic_1)
        return pic_1, pic_2


#连续两次卷积
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, c_mid=None):
        super().__init__()
        if not c_mid:
            c_mid = c_out
        self.double_conv = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 3, 1, 1),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, 1, 1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#下采样，通过一个池化层再接一个DoubleConv
class DownSampling(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(c_in, c_out)
        )

    def forward(self, x):
        return self.downsampling(x)


#上采样，同时进行特征融合
class UpSampling(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.upsampling = nn.ConvTranspose2d(c_in, c_in // 2, 2, 2)  #反卷积
        self.conv = DoubleConv(c_in, c_out)

    def forward(self, x1, x2):
        x1 = self.upsampling(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


#根据分割数量，整合输出通道
class OutConv(nn.Module):
    def __init__(self, c_in, c_out):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        return self.conv(x)


#定义网络结构
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inconv = DoubleConv(n_channels, 64)
        self.downsampling1 = DownSampling(64, 128)   #连续4个下采样
        self.downsampling2 = DownSampling(128, 256)
        self.downsampling3 = DownSampling(256, 512)
        self.downsampling4 = DownSampling(512, 1024)
        self.upsampling1 = UpSampling(1024, 512)     #连续4个上采样
        self.upsampling2 = UpSampling(512, 256)
        self.upsampling3 = UpSampling(256, 128)
        self.upsampling4 = UpSampling(128, 64)
        self.outconv = OutConv(64, n_classes)        #输出卷积

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.downsampling1(x1)
        x3 = self.downsampling2(x2)
        x4 = self.downsampling3(x3)
        x5 = self.downsampling4(x4)
        x = self.upsampling1(x5, x4)
        x = self.upsampling2(x, x3)
        x = self.upsampling3(x, x2)
        x = self.upsampling4(x, x1)
        out = self.outconv(x)
        return out


#miou的计算
def Miou(pred, target):
    miou = 0
    pred = torch.argmax(pred, 1)
    B, W, H = pred.shape          #预测图片的长宽高
    for i in range(B):
        predict = pred[i]
        mask = target[i]
        union = torch.logical_or(predict, mask).sum()
        if union < 1e-5:
            return 0
        inter = ((predict + mask) == 2).sum()
        miou += inter / union
    return miou / B


#得到二进制mask的边界
def Boundary(pic, _mask):
    if not _mask:
        pic = torch.argmax(pic, 1).cpu().numpy().astype('float64')
    else:
        pic = pic.cpu().numpy()
    B, W, H = pic.shape
    new_pic = np.zeros([B, W + 2, H + 2])
    mask_erode = np.zeros([B, W, H])
    dilation = int(round(0.02 * np.sqrt(W ** 2 + H ** 2)))  #对角线长度，
    if dilation < 1:
        dilation = 1
    for i in range(B):
        new_pic[i] = cv2.copyMakeBorder(pic[i], 1, 1, 1, 1,
                                        cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for j in range(B):
        erode = cv2.erode(new_pic[j], kernel, iterations=dilation)
        mask_erode[j] = erode[1: W + 1, 1: H + 1]
    return torch.from_numpy(pic - mask_erode)


#计算biou
def Biou(pred, target):
    intersection = 0
    union = 0
    pred = Boundary(pred, _mask=False)
    target = Boundary(target, _mask=True)
    B, W, H = pred.shape
    for i in range(B):
        predict = pred[i]
        mask = target[i]
        intersection += ((predict * mask) > 0).sum()
        union += ((predict + mask) > 0).sum()
    if union < 1:
        return 0
    biou = (intersection / union)
    return biou





# 实例化数据集
data = myDataset(transform)
train_size = int(0.85 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = random_split(data, [train_size, test_size])

# 利用DataLoader生成一个分batch获取数据的可迭代对象
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

epochs = 30
model = UNet(3, 2)                #通过unet
loss = nn.CrossEntropyLoss()      #计算交叉熵
opt = optim.Adam(model.parameters(), lr=1e-5)#adam优化
miou = []
biou = []

for epoch in range(epochs):
    for X, y in train_loader:
        train_loss = []
        opt.zero_grad()
        l = loss(model(X), y)
        train_loss.append(l)
        l.backward()
        opt.step()
    train_loss = sum(train_loss)
    print('epoch:', epoch + 1, 'loss:', train_loss)

    test_miou = []
    test_biou = []
    with torch.no_grad():
        test_loss = []
        for X, y in test_loader:
            l = loss(model(X), y)
            test_loss.append(l)
        test_loss = sum(test_loss)
        print('test_loss:', test_loss)
        test_miou.append(Miou(model(X), y))
        test_biou.append(Biou(model(X), y))
    miou.append(sum(test_miou) / len(test_miou))
    biou.append(sum(test_biou) / len(test_biou))
    print('miou:', miou)
    print('biou:', biou)
    if epoch>=10:
        torch.save(model.state_dict(), 'checkpoints/unet_model_{}.pth'.format(epoch))

