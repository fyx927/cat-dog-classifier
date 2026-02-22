import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import get_data_loaders  # 导入我们自己写的数据加载函数


if __name__ == '__main__':
    # 加载数据
    train_loader, val_loader, class_names = get_data_loaders(data_dir='train', batch_size=32)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 模型定义
    model = models.resnet18(pretrained=True)

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层全连接层，输出为2类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))  # class_names = ['cats','dogs']
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 训练函数
    def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
        since = time.time()
        train_loss_history = []
        val_acc_history = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)

            # 每个epoch有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # 遍历数据
                for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                else:
                    val_acc_history.append(epoch_acc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(f'\n训练完成，总用时 {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'最佳验证准确率: {best_acc:.4f}')

        # 加载最佳模型权重
        model.load_state_dict(best_model_wts)
        return model, train_loss_history, val_acc_history

    # 执行训练
    model, train_loss, val_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)

    # 保存模型
    torch.save(model.state_dict(), 'cat_dog_resnet18.pth')
    print("模型已保存为 cat_dog_resnet18.pth")

    # 绘制训练曲线
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss, 'b-o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(1,2,2)
    val_acc_cpu = [acc.cpu().numpy() * 100 for acc in val_acc]  # 转换为百分数
    plt.plot(range(1, len(val_acc_cpu)+1), val_acc_cpu, 'r-o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("训练曲线已保存为 training_curves.png")
