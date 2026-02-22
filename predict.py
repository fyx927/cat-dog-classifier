import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 重新构建模型
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2分类
model.load_state_dict(torch.load('cat_dog_resnet18.pth', map_location=device))
model = model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 输入图片路径，返回预测类别
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
    return 'dog' if pred.item() == 1 else 'cat'


# 从训练集中随机挑选几张猫和狗图片进行测试
cat_images = [os.path.join('train/cats', f) for f in os.listdir('train/cats') if f.endswith('.jpg')]
dog_images = [os.path.join('train/dogs', f) for f in os.listdir('train/dogs') if f.endswith('.jpg')]

# 随机选各2张
num_samples = min(2, len(cat_images), len(dog_images))
test_images = random.sample(cat_images, num_samples) + random.sample(dog_images, num_samples)
true_labels = ['cat'] * num_samples + ['dog'] * num_samples

# 显示结果
plt.figure(figsize=(12, 8))
for i, (img_path, true_label) in enumerate(zip(test_images, true_labels)):
    pred_label = predict_image(img_path)
    color = 'green' if pred_label == true_label else 'red'

    img = Image.open(img_path)
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(f'True: {true_label} | Predict: {pred_label}', color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('prediction_results.png')
print("预测结果图已保存为 prediction_results.png")
