#!/usr/bin/env python3
"""
Generate Technical Report Visualizations
Run this AFTER training to create charts for the report.

Usage (on Kaggle):
    1. Run train_optimized.ipynb first
    2. Then run this script to generate all charts

Output files:
    - confusion_matrices.png
    - age_distribution.png
    - sample_predictions.png
    - augmentation_examples.png
    - per_class_accuracy.csv
    - final_summary.csv
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm
from PIL import Image
import cv2

# Import from training notebook (run notebook cells first)
# from train_optimized import cfg, model, create_datasets, get_train_transforms

# ==========================================
# Configuration (copy from notebook)
# ==========================================
class Config:
    OUTPUT_DIR = "/kaggle/working"
    IMG_SIZE = 160
    NUM_AGES = 10
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    AGE_GROUPS = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    STUDENT_BACKBONE = "mobilenetv3_large_100"

cfg = Config()

# ==========================================
# 14. Confusion Matrix Visualization
# ==========================================
def plot_confusion_matrices(all_gender_true, all_gender_preds, all_age_true, all_age_preds):
    """Generate and plot confusion matrices for gender and age"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gender Confusion Matrix
    cm_gender = confusion_matrix(all_gender_true, all_gender_preds)
    sns.heatmap(cm_gender, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
    axes[0].set_title('Gender Confusion Matrix', fontsize=14)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Age Confusion Matrix
    cm_age = confusion_matrix(all_age_true, all_age_preds)
    sns.heatmap(cm_age, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=cfg.AGE_GROUPS, yticklabels=cfg.AGE_GROUPS)
    axes[1].set_title('Age Group Confusion Matrix', fontsize=14)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print classification reports
    print("\n" + "="*50)
    print("GENDER CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_gender_true, all_gender_preds, target_names=['Male', 'Female']))
    
    print("\n" + "="*50)
    print("AGE GROUP CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_age_true, all_age_preds, target_names=cfg.AGE_GROUPS))


# ==========================================
# 15. Age Distribution Analysis
# ==========================================
def plot_age_distribution(age_true, age_preds):
    """Plot age group distribution analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Dataset distribution
    axes[0].bar(range(len(cfg.AGE_GROUPS)), np.bincount(age_true, minlength=10), color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(len(cfg.AGE_GROUPS)))
    axes[0].set_xticklabels(cfg.AGE_GROUPS, rotation=45)
    axes[0].set_title('Dataset Age Distribution', fontsize=12)
    axes[0].set_xlabel('Age Group')
    axes[0].set_ylabel('Count')
    
    # Predicted vs Actual
    x = np.arange(len(cfg.AGE_GROUPS))
    width = 0.35
    axes[1].bar(x - width/2, np.bincount(age_true, minlength=10), width, label='Actual', color='steelblue')
    axes[1].bar(x + width/2, np.bincount(age_preds, minlength=10), width, label='Predicted', color='coral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cfg.AGE_GROUPS, rotation=45)
    axes[1].set_title('Predicted vs Actual Distribution', fontsize=12)
    axes[1].legend()
    
    # Error distribution
    errors = np.array(age_preds) - np.array(age_true)
    axes[2].hist(errors, bins=range(-5, 6), color='salmon', edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='green', linestyle='--', linewidth=2)
    axes[2].set_title('Prediction Error Distribution', fontsize=12)
    axes[2].set_xlabel('Error (Predicted - Actual groups)')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/age_distribution.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate ±1 accuracy
    within_1 = np.mean(np.abs(errors) <= 1) * 100
    print(f"\n📊 Age Estimation Metrics:")
    print(f"   Exact match: {np.mean(np.array(age_preds) == np.array(age_true))*100:.1f}%")
    print(f"   Within ±1 group: {within_1:.1f}%")
    print(f"   Mean Absolute Error: {np.mean(np.abs(errors)):.2f} groups")


# ==========================================
# 16. Per-Class Accuracy Report
# ==========================================
def per_class_accuracy_report(gender_true, gender_preds, age_true, age_preds):
    """Generate detailed per-class accuracy report"""
    
    gender_true = np.array(gender_true)
    gender_preds = np.array(gender_preds)
    
    male_acc = np.mean(gender_preds[gender_true == 0] == 0) * 100
    female_acc = np.mean(gender_preds[gender_true == 1] == 1) * 100
    
    print("="*50)
    print("GENDER ACCURACY BREAKDOWN")
    print("="*50)
    print(f"  Male accuracy:   {male_acc:.1f}%")
    print(f"  Female accuracy: {female_acc:.1f}%")
    print(f"  Overall:         {np.mean(gender_preds == gender_true)*100:.1f}%")
    
    age_true = np.array(age_true)
    age_preds = np.array(age_preds)
    
    print("\n" + "="*50)
    print("AGE GROUP ACCURACY BREAKDOWN")
    print("="*50)
    print(f"{'Age Group':<12} {'Count':<8} {'Exact':<10} {'±1 Group':<10}")
    print("-"*40)
    
    results = []
    for i, group in enumerate(cfg.AGE_GROUPS):
        mask = age_true == i
        count = np.sum(mask)
        if count > 0:
            exact = np.mean(age_preds[mask] == i) * 100
            within_1 = np.mean(np.abs(age_preds[mask] - i) <= 1) * 100
            print(f"{group:<12} {count:<8} {exact:>6.1f}%   {within_1:>6.1f}%")
            results.append({'Age Group': group, 'Count': count, 'Exact Acc': exact, '±1 Acc': within_1})
    
    pd.DataFrame(results).to_csv(f"{cfg.OUTPUT_DIR}/per_class_accuracy.csv", index=False)
    print(f"\n✅ Saved: {cfg.OUTPUT_DIR}/per_class_accuracy.csv")


# ==========================================
# 17. Sample Predictions Visualization
# ==========================================
def plot_sample_predictions(model, val_dataset, device, num_samples=16):
    """Show grid of sample predictions with confidence"""
    model.eval()
    
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()
    
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    for idx, ax in zip(indices, axes):
        img, gender_true, age_true, _ = val_dataset[idx]
        
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            gender_logits, age_logits = model(img_tensor)
            
            gender_probs = F.softmax(gender_logits, dim=1)
            age_probs = F.softmax(age_logits, dim=1)
            
            gender_pred = gender_logits.argmax(1).item()
            age_pred = age_logits.argmax(1).item()
            gender_conf = gender_probs.max().item()
            age_conf = age_probs.max().item()
        
        # Denormalize image for display
        img_display = img.numpy().transpose(1, 2, 0)
        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_display = np.clip(img_display, 0, 1)
        
        ax.imshow(img_display)
        
        gender_correct = gender_pred == gender_true
        age_correct = age_pred == age_true
        
        gender_str = ['Male', 'Female'][gender_pred]
        age_str = cfg.AGE_GROUPS[age_pred]
        
        color = 'green' if (gender_correct and age_correct) else 'red'
        ax.set_title(f"{gender_str} ({gender_conf:.0%})\n{age_str} ({age_conf:.0%})", 
                     fontsize=10, color=color)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.show()


# ==========================================
# 18. Model Architecture Summary
# ==========================================
def model_summary(model):
    """Print model summary with parameter counts"""
    print("="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter Count:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable:        {total_params - trainable_params:,}")
    
    model_size = total_params * 4 / (1024 * 1024)
    print(f"\n💾 Model Size (FP32): {model_size:.2f} MB")
    print(f"   Model Size (INT8):  {model_size/4:.2f} MB (estimated)")
    
    print(f"\n🔧 Layer Breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"   {name:<20}: {params:>10,} params")
    
    print(f"\n📈 Comparison with Teacher (EfficientNetV2-S):")
    print(f"   Teacher params: ~21,000,000")
    print(f"   Student params: {total_params:,}")
    print(f"   Compression:    {21000000/total_params:.1f}x smaller")


# ==========================================
# 20. Final Summary Table
# ==========================================
def generate_final_summary(model, history, gender_true, gender_preds, age_true, age_preds):
    """Generate comprehensive final summary"""
    
    gender_acc = np.mean(np.array(gender_preds) == np.array(gender_true)) * 100
    age_exact = np.mean(np.array(age_preds) == np.array(age_true)) * 100
    age_within_1 = np.mean(np.abs(np.array(age_preds) - np.array(age_true)) <= 1) * 100
    age_mae = np.mean(np.abs(np.array(age_preds) - np.array(age_true)))
    
    total_params = sum(p.numel() for p in model.parameters())
    
    summary = {
        'Metric': [
            'Gender Accuracy',
            'Age Exact Accuracy', 
            'Age ±1 Group Accuracy',
            'Age MAE (groups)',
            'Model Parameters',
            'Model Size (FP32)',
            'Backbone',
            'Input Size',
            'Training Epochs',
        ],
        'Value': [
            f'{gender_acc:.1f}%',
            f'{age_exact:.1f}%',
            f'{age_within_1:.1f}%',
            f'{age_mae:.2f}',
            f'{total_params:,}',
            f'{total_params * 4 / 1e6:.2f} MB',
            cfg.STUDENT_BACKBONE,
            f'{cfg.IMG_SIZE}x{cfg.IMG_SIZE}',
            str(len(history) if history else 'N/A'),
        ]
    }
    
    df = pd.DataFrame(summary)
    
    print("\n" + "="*60)
    print("📊 FINAL MODEL SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    
    df.to_csv(f"{cfg.OUTPUT_DIR}/final_summary.csv", index=False)
    print(f"\n✅ Saved: {cfg.OUTPUT_DIR}/final_summary.csv")


# ==========================================
# Main: Run all visualizations
# ==========================================
def run_all_visualizations(model, val_loader, val_dataset, device, history=None):
    """Run all visualization functions"""
    print("\n" + "="*60)
    print("🎨 GENERATING ALL VISUALIZATIONS FOR REPORT")
    print("="*60)
    
    # Collect predictions
    print("\n1. Collecting predictions...")
    model.eval()
    all_gender_preds, all_gender_true = [], []
    all_age_preds, all_age_true = [], []
    
    with torch.no_grad():
        for images, genders, age_groups, _ in tqdm(val_loader, desc="Predicting"):
            images = images.to(device)
            pred_gender, pred_age = model(images)
            all_gender_preds.extend(pred_gender.argmax(1).cpu().numpy())
            all_gender_true.extend(genders.numpy())
            all_age_preds.extend(pred_age.argmax(1).cpu().numpy())
            all_age_true.extend(age_groups.numpy())
    
    # Generate visualizations
    print("\n2. Generating confusion matrices...")
    plot_confusion_matrices(all_gender_true, all_gender_preds, all_age_true, all_age_preds)
    
    print("\n3. Generating age distribution plots...")
    plot_age_distribution(all_age_true, all_age_preds)
    
    print("\n4. Generating per-class accuracy report...")
    per_class_accuracy_report(all_gender_true, all_gender_preds, all_age_true, all_age_preds)
    
    print("\n5. Generating sample predictions grid...")
    plot_sample_predictions(model, val_dataset, device)
    
    print("\n6. Generating model summary...")
    model_summary(model)
    
    print("\n7. Generating final summary...")
    generate_final_summary(model, history, all_gender_true, all_gender_preds, all_age_true, all_age_preds)
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nOutput files in:", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    print("This script should be run from Kaggle notebook after training.")
    print("Import the trained model and call: run_all_visualizations(model, val_loader, val_dataset, device, history)")
