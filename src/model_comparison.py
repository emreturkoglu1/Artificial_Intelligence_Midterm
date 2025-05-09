"""
Model Comparison Script

This script compares the performance of different models trained on the flower
classification task. It generates comparative visualizations and metrics.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_metrics_file(file_path):
    """
    Read metrics from a file.

    Args:
        file_path (str): Path to the metrics file

    Returns:
        dict: Dictionary containing the metrics
    """
    metrics = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Get model name
        metrics['Model'] = lines[0].split(': ')[1].strip()
        
        # Get other metrics
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(': ')
                key = key.strip()
                
                # Special handling for Training Time to remove " seconds"
                if key == "Training Time":
                    value = value.replace("seconds", "").strip()
                
                try:
                    # Convert numerical values to float
                    metrics[key] = float(value.strip())
                except ValueError:
                    # Keep string values as is
                    metrics[key] = value.strip()
    
    return metrics


def plot_performance_comparison(metrics_df, output_dir):
    """
    Create performance comparison plots.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing model metrics
        output_dir (str): Directory to save the plots
    """
    # Ensure metrics directory exists
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convert numeric columns to float
    numeric_cols = ['Training Time', 'Best Validation Accuracy', 'Final Training Loss', 
                    'Final Training Accuracy', 'Final Validation Loss', 'Final Validation Accuracy']
    
    for col in numeric_cols:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
    
    # Accuracy comparison
    sns.barplot(x='Model', y='Best Validation Accuracy', data=metrics_df, ax=axes[0, 0])
    axes[0, 0].set_title('Validation Accuracy Comparison')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Training time comparison
    sns.barplot(x='Model', y='Training Time', data=metrics_df, ax=axes[0, 1])
    axes[0, 1].set_title('Training Time Comparison (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Training loss comparison
    sns.barplot(x='Model', y='Final Training Loss', data=metrics_df, ax=axes[1, 0])
    axes[1, 0].set_title('Final Training Loss Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Training accuracy comparison
    sns.barplot(x='Model', y='Final Training Accuracy', data=metrics_df, ax=axes[1, 1])
    axes[1, 1].set_title('Final Training Accuracy Comparison')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'model_comparison.png'))
    plt.close()


def create_comparison_table(metrics_df, output_dir):
    """
    Create a comparison table in CSV format.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing model metrics
        output_dir (str): Directory to save the table
    """
    # Ensure metrics directory exists
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Also create a markdown file with the comparison
    md_path = os.path.join(metrics_dir, 'model_comparison.md')
    with open(md_path, 'w') as f:
        f.write('# Model Comparison\n\n')
        f.write('| Model | Training Time (s) | Best Val Acc | Train Loss | Train Acc | Val Loss | Val Acc |\n')
        f.write('|-------|-----------------|-------------|-----------|----------|----------|--------|\n')
        
        for _, row in metrics_df.iterrows():
            f.write(f"| {row['Model']} | {row['Training Time']:.2f} | {row['Best Validation Accuracy']:.4f} | ")
            f.write(f"{row['Final Training Loss']:.4f} | {row['Final Training Accuracy']:.4f} | ")
            f.write(f"{row['Final Validation Loss']:.4f} | {row['Final Validation Accuracy']:.4f} |\n")
    
    # Select relevant columns
    columns = [
        'Model',
        'Training Time',
        'Best Validation Accuracy',
        'Final Training Loss',
        'Final Training Accuracy',
        'Final Validation Loss',
        'Final Validation Accuracy'
    ]
    
    # Create comparison table
    comparison_table = metrics_df[columns].copy()
    
    # Round numerical values
    for col in columns[1:]:
        comparison_table[col] = comparison_table[col].round(4)
    
    # Save to CSV
    comparison_table.to_csv(os.path.join(metrics_dir, 'model_comparison.csv'), 
                          index=False)


def main(args):
    """
    Main function for model comparison.

    Args:
        args: Parsed command line arguments
    """
    # Ensure metrics directory exists
    metrics_dir = os.path.join(args.output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # List of metric files to compare
    metric_files = [
        'custom_cnn_metrics.txt',
        'vgg16_feature_extractor_metrics.txt',
        'vgg16_fine_tuned_metrics.txt'
    ]
    
    # Read metrics from all files
    metrics_list = []
    for file_name in metric_files:
        file_path = os.path.join(metrics_dir, file_name)
        if os.path.exists(file_path):
            metrics = read_metrics_file(file_path)
            metrics_list.append(metrics)
    
    if not metrics_list:
        print("No metric files found in the metrics directory")
        return
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Create visualizations
    plot_performance_comparison(metrics_df, args.output_dir)
    
    # Create comparison table
    create_comparison_table(metrics_df, args.output_dir)
    
    print("Model comparison completed. Results saved in the metrics directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Model Performance')
    
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory containing model metrics and to save comparisons')
    
    args = parser.parse_args()
    main(args) 