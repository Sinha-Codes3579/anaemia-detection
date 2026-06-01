import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# add project path
sys.path.append('/content/drive/MyDrive/anaemiadetect/Anaemia')

from config.config import Config
from utils.data_loader import DataManager
from models.anaemia_model import create_model

def analyze_predictions():
    """Simple analysis to understand model behavior"""
    print("="*60)
    print("SIMPLIFIED MODEL ANALYSIS")
    print("="*60)
    
    # Load config
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    model = create_model('efficientnet', num_classes=config.NUM_CLASSES)
    
    # Find the latest model file
    model_dir = config.MODEL_DIR
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and 'efficientnet' in f]
    
    if not model_files:
        print(f"❌ No model files found in {model_dir}")
        print("Available files:", os.listdir(model_dir))
        return
    
    # Use the first model file
    model_path = os.path.join(model_dir, model_files[0])
    print(f"📁 Loading model from: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load data
    print("\n📊 Loading validation dataset...")
    data_manager = DataManager(config)
    data_manager.discover_data()
    _, val_loader, test_loader = data_manager.get_data_loaders()
    
    # Get a batch of validation data
    batch_images = []
    batch_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            batch_images = images
            batch_labels = labels
            break  # Just get first batch
    
    print(f"📸 Analyzing {len(batch_images)} validation images...")
    
    # Get predictions and confidence scores
    predictions = []
    confidences = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(len(batch_images)):
            image = batch_images[i].unsqueeze(0).to(device)
            output = model(image)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
            
            predictions.append(pred.item())
            confidences.append(confidence.item())
            all_probs.append(probs.cpu().numpy()[0])
    
    # Convert to numpy
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    batch_labels = batch_labels.numpy()
    
    # Calculate accuracy on this batch
    correct = (predictions == batch_labels).sum()
    accuracy = 100 * correct / len(batch_labels)
    
    print(f"\n✅ Batch Accuracy: {accuracy:.2f}% ({correct}/{len(batch_labels)} correct)")
    
    # 1. Check confidence distribution
    print(f"\n📊 Confidence Statistics:")
    print(f"   Average confidence: {confidences.mean():.4f}")
    print(f"   Min confidence: {confidences.min():.4f}")
    print(f"   Max confidence: {confidences.max():.4f}")
    print(f"   Std confidence: {confidences.std():.4f}")
    
    # 2. Check if model is overconfident
    high_confidence = (confidences > 0.99).sum()
    print(f"   Samples with >99% confidence: {high_confidence}/{len(confidences)} ({100*high_confidence/len(confidences):.1f}%)")
    
    # 3. Visualize predictions
    n_samples = min(12, len(batch_images))
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(n_samples):
        # Get image (unnormalized)
        img_np = batch_images[i].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Display
        axes[i].imshow(img_np)
        
        true_label = config.CLASS_NAMES[batch_labels[i]]
        pred_label = config.CLASS_NAMES[predictions[i]]
        confidence = confidences[i]
        
        # Color code: green if correct, red if wrong
        color = 'green' if predictions[i] == batch_labels[i] else 'red'
        
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}", 
                         color=color, fontsize=9)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Model Predictions on Validation Batch (Accuracy: {accuracy:.1f}%)", fontsize=14)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(config.RESULTS_DIR, 'simple_predictions_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: {output_path}")
    plt.show()
    
    # 4. Check if model is using trivial features by analyzing image statistics
    print("\n" + "="*60)
    print("IMAGE STATISTICS ANALYSIS (Check for trivial features)")
    print("="*60)
    
    # Analyze average brightness per class
    anaemic_brightness = []
    non_anaemic_brightness = []
    
    for i in range(len(batch_images)):
        img = batch_images[i]
        # Convert to grayscale brightness
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        avg_brightness = gray.mean().item()
        
        if batch_labels[i] == 0:  # Anaemic
            anaemic_brightness.append(avg_brightness)
        else:  # Non-anaemic
            non_anaemic_brightness.append(avg_brightness)
    
    if anaemic_brightness and non_anaemic_brightness:
        print(f"📈 Average brightness (normalized 0-1 scale):")
        print(f"   Anaemic images: {np.mean(anaemic_brightness):.4f} ± {np.std(anaemic_brightness):.4f}")
        print(f"   Non-anaemic images: {np.mean(non_anaemic_brightness):.4f} ± {np.std(non_anaemic_brightness):.4f}")
        
        # T-test to check if brightness is significantly different
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(anaemic_brightness, non_anaemic_brightness)
        print(f"   Statistical difference (t-test): p = {p_value:.6f}")
        
        if p_value < 0.05:
            print("   ✅ Classes have significantly different brightness (expected for nail color)")
        else:
            print("   ⚠️  Classes don't have significantly different brightness (unexpected)")
    
    # 5. Check model output distribution
    print("\n" + "="*60)
    print("MODEL OUTPUT DISTRIBUTION")
    print("="*60)
    
    # Collect all test predictions
    test_predictions = []
    test_confidences = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)
            
            test_predictions.extend(preds.cpu().numpy())
            test_confidences.extend(confidence.cpu().numpy())
    
    test_confidences = np.array(test_confidences)
    
    # Plot confidence histogram
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(test_confidences, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidences (Test Set)')
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Plot CDF
    sorted_conf = np.sort(test_confidences)
    cdf = np.arange(1, len(sorted_conf)+1) / len(sorted_conf)
    plt.plot(sorted_conf, cdf, linewidth=2)
    plt.xlabel('Confidence')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Confidences')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    conf_path = os.path.join(config.RESULTS_DIR, 'confidence_distribution.png')
    plt.savefig(conf_path, dpi=150, bbox_inches='tight')
    print(f"💾 Confidence distribution saved to: {conf_path}")
    plt.show()
    
    print(f"\n📊 Test Set Confidence Statistics:")
    print(f"   Mean: {test_confidences.mean():.4f}")
    print(f"   Median: {np.median(test_confidences):.4f}")
    print(f"   % with confidence > 0.99: {100*(test_confidences > 0.99).sum()/len(test_confidences):.1f}%")
    print(f"   % with confidence > 0.95: {100*(test_confidences > 0.95).sum()/len(test_confidences):.1f}%")
    print(f"   % with confidence < 0.70: {100*(test_confidences < 0.70).sum()/len(test_confidences):.1f}%")
    
    # 6. Critical check: Find ANY uncertain predictions
    uncertain_indices = np.where(test_confidences < 0.8)[0]
    if len(uncertain_indices) > 0:
        print(f"\n🔍 Found {len(uncertain_indices)} uncertain predictions (confidence < 80%)")
        print("   These are the most valuable for understanding model limitations.")
    else:
        print(f"\n⚠️  WARNING: ALL predictions have >80% confidence")
        print("   This is statistically improbable and suggests:")
        print("   1. Dataset is too easy/classes are trivially separable")
        print("   2. Data leakage between train/val/test")
        print("   3. Model is overfitting to trivial features")
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    # Final diagnosis
    if accuracy == 100 and test_confidences.mean() > 0.99:
        print("❓ STATISTICALLY IMPROBABLE RESULTS DETECTED")
        print("\nRECOMMENDED ACTIONS:")
        print("1. Run data leakage check: python check_data_leakage.py")
        print("2. Manually inspect 20 random images from each class")
        print("3. Check if filenames contain class information")
        print("4. Verify train/val/test splits are truly independent")
        print("\n⚠️  Do NOT present 100% accuracy without thorough validation!")
    else:
        print("✅ Results appear reasonable")
        print(f"   Validation accuracy: {accuracy:.2f}%")
        print(f"   Average confidence: {test_confidences.mean():.2%}")

if __name__ == "__main__":
    analyze_predictions()