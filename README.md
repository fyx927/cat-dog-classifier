# Cats vs Dogs Image Classifier

A transfer learning based image classifier for cats and dogs using PyTorch and ResNet18.  
Achieved **98.1% validation accuracy** on the Kaggle Dogs vs Cats dataset.

## Project Structure
- `train.py` - Training script
- `predict.py` - Prediction script
- `data_loader.py` - Data loading and preprocessing
- `training_curves.png` - Training loss and accuracy curves
- `prediction_results.png` - Sample prediction results
- `cat_dog_resnet18.pth` - Trained model weights (not included in this repo due to size)
- `requirements.txt` - List of dependencies

## Requirements
- Python 3.9
- PyTorch (CPU version is sufficient)
- torchvision
- matplotlib
- numpy
- tqdm
- pillow

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Dataset

Download the [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle.
Extract `train.zip` and organize the images as follows:

```text
train/
  cats/    # all cat images
  dogs/    # all dog images
```

## Training

Run the training script:

```bash
python train.py
```

The script will:

- Load and preprocess data
- Fine-tune a pretrained ResNet18
- Save the best model as `cat_dog_resnet18.pth`
- Generate training curves `training_curves.png`

## Prediction

To test the model on random images from the training set:

```bash
python predict.py
```

This will create `prediction_results.png` showing true vs. predicted labels.

## Results

- **Best validation accuracy**: 98.1%

[https://training_curves.png](https://training_curves.png/)

https://prediction_results.png