# Flower Classification Project

This project aims to classify flower images using different deep learning models.

## Author
**Name:** Emre TÜrkoğlu  
**Student ID:** 210717046  
**Department:** Software Engineering  
**University:** Muğla Sıtkı Koçman University  
**Course:** SE3508 Introduction to Artificial Intelligence

## Project Structure

```
.
├── data/                   # Dataset directory
│   └── flowers/           # Flower images
├── src/                   # Source code
│   ├── models/           # Model definitions
│   │   ├── custom_cnn.py     # Custom CNN model
│   │   └── vgg_models.py     # VGG16 based models
│   ├── utils/            # Helper functions
│   │   ├── data_loader.py    # Data loading operations
│   │   └── trainer.py        # Model training functions
│   ├── visualizations/   # Visualization tools
│   │   └── feature_visualization.py
│   ├── training/        # Training scripts
│   │   ├── train_custom_cnn.py           # Custom CNN training
│   │   ├── train_vgg16_feature_extractor.py  # VGG16 feature extractor training
│   │   └── train_vgg16_fine_tuned.py     # VGG16 fine tuning training
│   └── model_comparison.py           # Model comparison
└── output/               # Output directory
    ├── models/          # Trained models
    ├── visualizations/  # Visualizations
    └── metrics/         # Performance metrics

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
- Download the [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset/data)
- Extract it to the `data/` directory

## Usage

### Training the Custom CNN Model
```bash
python src/training/train_custom_cnn.py --data_dir ./data --output_dir ./output --batch_size 64 --epochs 35
```

### Training the VGG16 Feature Extractor Model
```bash
python src/training/train_vgg16_feature_extractor.py --data_dir ./data --output_dir ./output --batch_size 64 --epochs 35
```

### Training the VGG16 Fine-Tuned Model
```bash
python src/training/train_vgg16_fine_tuned.py --data_dir ./data --output_dir ./output --batch_size 64 --epochs 35
```

### Model Comparison
```bash
python src/model_comparison.py --output_dir ./output
```

### Making Predictions with Trained Models
```bash
# Using VGG16 Fine-Tuned model (single prediction)
python src/predict_random.py --model_path output/models/vgg16_fine_tuned_best.pth --model_type vgg16_fine_tuned

# Using VGG16 Fine-Tuned model (multiple predictions)
python src/predict_random.py --model_path output/models/vgg16_fine_tuned_best.pth --model_type vgg16_fine_tuned --multiple --num_predictions 20

# Using VGG16 Feature Extractor model
python src/predict_random.py --model_path output/models/vgg16_feature_extractor_best.pth --model_type vgg16_feature_extractor

# Using Custom CNN model
python src/predict_random.py --model_path output/models/custom_cnn_best.pth --model_type custom_cnn
```

The prediction script automatically selects random images from the training dataset and shows:
- True class
- Predicted class
- Prediction probability
- Visual result (✓ for correct, ✗ for wrong)

When using `--multiple`, it also shows:
- Overall accuracy statistics
- Per-class performance metrics
- Detailed results for each prediction

## Parameters

Common parameters for each training script:

- `--data_dir`: Dataset directory (default: './data')
- `--output_dir`: Output directory (default: './output')
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 25)
- `--learning_rate`: Learning rate (default: 0.00001)
- `--weight_decay`: Weight decay (default: 1e-3)
- `--num_workers`: Number of data loading workers (default: 4)
- `--no_cuda`: Disable GPU usage (default: False)

## Model Architectures

### Custom CNN
- 5 convolutional blocks
- Each block: Convolution, Batch Normalization, ReLU, MaxPooling
- 3 fully connected layers
- Dropout layers (0.7)

### VGG16 Feature Extractor
- Pre-trained VGG16 convolutional layers (frozen)
- Custom classifier layers

### VGG16 Fine-Tuned
- Pre-trained VGG16 (first block frozen)
- Fine-tuned convolutional layers
- Custom classifier layers

## Performance Metrics

Models are evaluated on the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Training Time

## Visualizations

Visualizations created for each model:
- Training/validation loss graphs
- Training/validation accuracy graphs
- Feature maps
- Zeiler & Fergus visualizations

## Dataset

This project uses the [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset/data) from Kaggle. The dataset contains flowers in five categories:

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

## References

[1] Zeiler, M. D., & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks.

[2] Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.

## License

This project was developed as an assignment for the SE3508 Introduction to Artificial Intelligence course taught by Dr. Selim Yılmaz.

This project was completed as part of the SE3508 Introduction to Artificial Intelligence course, instructed by Dr. Selim Yılmaz, Department of Software Engineering at Muğla Sıtkı Koçman University, 2025.

**Note**: This repository must not be used by students in the same faculty in future years—whether partially or fully—as their own submission. Any form of code reuse without proper modification and original contribution will be considered by the instructor a violation of academic integrity policies.
