"""
Random Flower Prediction Script

This script selects a random flower image from the dataset and makes predictions using the trained model.
"""

import os
import random
import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.vgg_models import VGG16FineTuned, VGG16FeatureExtractor
from models.custom_cnn import CustomCNN

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def get_random_image(data_dir):
    """
    Select a random image from the dataset.
    
    Args:
        data_dir (str): Dataset directory
        
    Returns:
        tuple: (Image path, true class)
    """
    # Set up the correct path according to data directory structure
    train_dir = os.path.join(data_dir, 'train')
    
    # List class directories
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # Select a random class
    random_class = random.choice(class_dirs)
    
    # List all images in the selected class directory
    class_path = os.path.join(train_dir, random_class)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select a random image
    random_image = random.choice(image_files)
    
    return os.path.join(class_path, random_image), random_class

def load_model(model_path, model_type='vgg16_fine_tuned'):
    """
    Load the trained model.
    
    Args:
        model_path (str): Path to the saved model
        model_type (str): Type of model to load ('vgg16_fine_tuned', 'vgg16_feature_extractor', or 'custom_cnn')
    
    Returns:
        model: Loaded PyTorch model
    """
    # Determine which model architecture to use
    if model_type == 'vgg16_fine_tuned':
        model = VGG16FineTuned(num_classes=len(class_names))
    elif model_type == 'vgg16_feature_extractor':
        model = VGG16FeatureExtractor(num_classes=len(class_names))
    else:
        model = CustomCNN(num_classes=len(class_names))
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image_path, model):
    """
    Make a prediction for an image.
    
    Args:
        image_path (str): Path to the image file
        model: Trained PyTorch model
    
    Returns:
        tuple: Predicted class name and probability
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_prob, predicted_idx = torch.max(probabilities, 1)
    
    return class_names[predicted_idx], predicted_prob.item()

def display_prediction(image_path, true_class, predicted_class, probability):
    """
    Display the image with its prediction.
    
    Args:
        image_path (str): Path to the image file
        true_class (str): True class name
        predicted_class (str): Predicted class name
        probability (float): Prediction probability
    """
    # Load and display the image
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    
    # Check if prediction is correct
    is_correct = true_class == predicted_class
    result_text = "CORRECT ✓" if is_correct else "WRONG ✗"
    color = "green" if is_correct else "red"
    
    plt.title(f"True: {true_class}\nPredicted: {predicted_class}\nProbability: {probability:.2%}\n{result_text}", 
              color=color)
    plt.show()

def evaluate_multiple_predictions(model, data_dir, num_predictions=20):
    """
    Make predictions on multiple random images and show statistics.
    
    Args:
        model: Trained model
        data_dir: Dataset directory
        num_predictions: Number of predictions to make
    """
    correct = 0
    predictions = []
    
    print(f"\nTesting on {num_predictions} different images...")
    print("-" * 50)
    
    for i in range(num_predictions):
        # Select random image
        image_path, true_class = get_random_image(data_dir)
        
        # Make prediction
        predicted_class, probability = predict_image(image_path, model)
        
        # Save result
        is_correct = true_class == predicted_class
        if is_correct:
            correct += 1
            
        predictions.append({
            'image': os.path.basename(image_path),
            'true': true_class,
            'predicted': predicted_class,
            'probability': probability,
            'correct': is_correct
        })
        
        # Show brief info for each prediction
        print(f"Prediction {i+1:2d}: {'✓' if is_correct else '✗'} "
              f"(True: {true_class:10s} | Predicted: {predicted_class:10s} | "
              f"Probability: {probability:.2%})")
    
    # Show overall statistics
    accuracy = correct / num_predictions
    print("\nOverall Statistics:")
    print("-" * 50)
    print(f"Total Predictions: {num_predictions}")
    print(f"Correct Predictions: {correct}")
    print(f"Wrong Predictions: {num_predictions - correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Per-class statistics
    print("\nPer-class Statistics:")
    print("-" * 50)
    for class_name in class_names:
        class_predictions = [p for p in predictions if p['true'] == class_name]
        if class_predictions:
            class_correct = len([p for p in class_predictions if p['correct']])
            class_total = len(class_predictions)
            print(f"{class_name:10s}: {class_correct}/{class_total} "
                  f"({class_correct/class_total:.2%})")
    
    return predictions

def main(args):
    """
    Main function for prediction.
    
    Args:
        args: Command line arguments
    """
    # Load model
    model = load_model(args.model_path, args.model_type)
    
    if args.multiple:
        # Make multiple predictions
        predictions = evaluate_multiple_predictions(model, args.data_dir, args.num_predictions)
    else:
        # Make single prediction
        image_path, true_class = get_random_image(args.data_dir)
        predicted_class, probability = predict_image(image_path, model)
        
        # Show results
        print(f"\nPrediction Results:")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"True Class: {true_class}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Probability: {probability:.2%}")
        
        # Check if prediction is correct
        if true_class == predicted_class:
            print("Result: CORRECT ✓")
        else:
            print("Result: WRONG ✗")
        
        # Display image with prediction
        if not args.no_display:
            display_prediction(image_path, true_class, predicted_class, probability)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Flower Prediction')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Dataset directory')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='vgg16_fine_tuned',
                        choices=['vgg16_fine_tuned', 'vgg16_feature_extractor', 'custom_cnn'],
                        help='Type of model to use')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images')
    parser.add_argument('--multiple', action='store_true',
                        help='Make multiple predictions')
    parser.add_argument('--num_predictions', type=int, default=20,
                        help='Number of predictions to make')
    
    args = parser.parse_args()
    main(args) 