# Unet for Weizmann Horse Database

## 数据准备

1. 数据集： [Weizmann Horse Database | Kaggle](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database/metadata).
2. 该数据集包含327幅马的图像和掩蔽图像。
3. 训练验证集按0.85：0.15的比例随机划分

````
├──── horse(327 images)
│    ├──── horse001.png
│    ├──── horse002.png
│    └──── ...
├──── mask(327images)
│    ├──── horse001.png
│    ├──── horse002.png
│    └──...
````


## 训练

运行以下命令进行训练:

    python unet.py


## 测试 ##

测试集结果：MIoU： **0.919**  ,Boundary IoU： 0.731。


对应的命令：python unet.py


