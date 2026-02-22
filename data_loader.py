import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


# 数据预处理
def get_transform(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:  # validation
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_data_loaders(data_dir='train', batch_size=32, val_split=0.2, num_workers=2):
    """
    返回 train_loader, val_loader 以及数据集大小信息
    """
    
    full_dataset = datasets.ImageFolder(root=data_dir, transform=get_transform('train'))

    # 划分训练集和验证集
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 获取原数据集的样本列表
    targets = [full_dataset.imgs[i][1] for i in range(len(full_dataset))]
    # 构建两个数据集，分别使用不同的 transform
    train_dataset_v2 = datasets.ImageFolder(root=data_dir, transform=get_transform('train'))
    val_dataset_v2 = datasets.ImageFolder(root=data_dir, transform=get_transform('val'))

    # 使用相同的 indices 划分
    indices = list(range(len(full_dataset)))
    import numpy as np
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset_v2, train_indices)
    val_dataset = Subset(val_dataset_v2, val_indices)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"训练集大小: {len(train_dataset)} 图片")
    print(f"验证集大小: {len(val_dataset)} 图片")
    print(f"类别: {full_dataset.classes}")  # 应输出 ['cats', 'dogs']

    return train_loader, val_loader, full_dataset.classes
