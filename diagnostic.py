import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Add your project path
sys.path.append('/content/drive/MyDrive/anaemiadetect/Anaemia')

from config.config import Config
from utils.data_loader import DataManager
from models.anaemia_model import create_model

def load_model_and_data(model_type='efficientnet'):
    """Load your trained model and data"""
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load your trained model
    model = create_model(model_type, num_classes=config.NUM_CLASSES)
    
    # Update this path to your ACTUAL saved model
    model_path = config.MODEL_SAVE_PATH.replace('.pth', f'_{model_type}.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from: {model_path}")
    else:
        print(f"❌ Model not found at: {model_path}")
        print("Available models:")
        for f in os.listdir(config.MODEL_DIR):
            if f.endswith('.pth'):
                print(f"  - {f}")
        return None, None, None, None
    
    model.to(device)
    model.eval()
    
    # 2. Load data
    print("\n📊 Loading dataset...")
    data_manager = DataManager(config)
    data_manager.discover_data()
    train_loader, val_loader, test_loader = data_manager.get_data_loaders()
    
    return model, train_loader, val_loader, test_loader, device, config

def run_diagnostics():
    """Main diagnostic function"""
    print("="*60)
    print("ANEMIA DETECTION MODEL DIAGNOSTICS")
    print("="*60)
    
    # Load everything
    model, train_loader, val_loader, test_loader, device, config = load_model_and_data('efficientnet')
    
    if model is None:
        print("❌ Could not load model. Exiting.")
        return
    
    print(f"\n📈 Model: EfficientNet")
    print(f"📈 Classes: {config.CLASS_NAMES}")
    print(f"📈 Device: {device}")
    
    # Evaluate function
    def evaluate_loader(loader, loader_name):
        all_preds = []
        all_labels = []
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_samples += labels.size(0)
        
        accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"  {loader_name:12s}: {accuracy:.2f}% ({total_samples} samples)")
        
        return accuracy, all_preds, all_labels
    
    print("\n" + "="*60)
    print("ACCURACY ACROSS DATASETS")
    print("="*60)
    
    # Get accuracies
    train_acc, train_preds, train_labels = evaluate_loader(train_loader, "Train")
    val_acc, val_preds, val_labels = evaluate_loader(val_loader, "Validation")
    test_acc, test_preds, test_labels = evaluate_loader(test_loader, "Test")
    
    # Calculate gaps
    train_val_gap = train_acc - val_acc
    val_test_gap = val_acc - test_acc
    
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    print(f"Training Accuracy:    {train_acc:.2f}%")
    print(f"Validation Accuracy:  {val_acc:.2f}%")
    print(f"Test Accuracy:        {test_acc:.2f}%")
    print(f"Train-Validation Gap: {train_val_gap:.2f}%")
    print(f"Validation-Test Gap:  {val_test_gap:.2f}%")
    
    # Diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if train_acc > 98 and train_val_gap > 15:
        print("❌ SEVERE OVERFITTING: Model memorized training data")
        print("   → Action needed: Add regularization, reduce model complexity")
    elif train_acc > 95 and train_val_gap > 10:
        print("⚠️  MODERATE OVERFITTING: Model doesn't generalize well")
        print("   → Action: Increase dropout, add data augmentation")
    elif train_val_gap < 5 and val_test_gap < 5:
        print("✅ GOOD GENERALIZATION: Model performs consistently")
        print("   → Your 99.37% might be valid if data is clean")
    else:
        print("🔍 UNUSUAL PATTERN: Check for data leakage")
        print("   → Verify no images appear in multiple splits")
    
    # Confusion Matrix for Validation Set
    print("\n" + "="*60)
    print("VALIDATION SET CONFUSION MATRIX")
    print("="*60)
    
    cm = confusion_matrix(val_labels, val_preds)
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"[[{cm[0,0]:4d}  {cm[0,1]:4d}]   ← True Anaemic")
    print(f" [{cm[1,0]:4d}  {cm[1,1]:4d}]]   ← True Non-Anaemic")
    print(f"\nClass Names: {config.CLASS_NAMES}")
    
    # Classification Report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT (Validation)")
    print("="*60)
    print(classification_report(val_labels, val_preds, 
                              target_names=config.CLASS_NAMES,
                              digits=4))
    
    # Create visualization
    create_visualization(train_acc, val_acc, test_acc, cm, config.CLASS_NAMES)
    
    # Save results
    save_results(train_acc, val_acc, test_acc, cm, config)
    
    print("\n✅ Diagnostics complete! Check 'diagnostics_results.png' for visualization.")

def create_visualization(train_acc, val_acc, test_acc, cm, class_names):
    """Create visualization of results"""
    plt.figure(figsize=(15, 5))
    
    # 1. Accuracy comparison
    plt.subplot(1, 3, 1)
    datasets = ['Train', 'Validation', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['blue', 'orange', 'green']
    
    bars = plt.bar(datasets, accuracies, color=colors)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Across Datasets')
    plt.ylim([0, 105])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # 2. Confusion Matrix heatmap
    plt.subplot(1, 3, 2)
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Validation)')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 3. Accuracy gaps
    plt.subplot(1, 3, 3)
    gaps = [train_acc - val_acc, val_acc - test_acc]
    gap_labels = ['Train-Val Gap', 'Val-Test Gap']
    gap_colors = ['red' if gap > 10 else 'orange' if gap > 5 else 'green' for gap in gaps]
    
    bars = plt.bar(gap_labels, gaps, color=gap_colors)
    plt.ylabel('Accuracy Difference (%)')
    plt.title('Generalization Gaps')
    plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Acceptable threshold')
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Problem threshold')
    
    # Add value labels
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{gap:.1f}%', ha='center', va='bottom')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('diagnostics_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_results(train_acc, val_acc, test_acc, cm, config):
    """Save results to a text file"""
    with open('diagnostics_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("ANEMIA DETECTION MODEL DIAGNOSTICS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model: EfficientNet\n")
        f.write(f"Classes: {config.CLASS_NAMES}\n\n")
        
        f.write("ACCURACY RESULTS:\n")
        f.write(f"  Training Accuracy:    {train_acc:.2f}%\n")
        f.write(f"  Validation Accuracy:  {val_acc:.2f}%\n")
        f.write(f"  Test Accuracy:        {test_acc:.2f}%\n")
        f.write(f"  Train-Validation Gap: {train_acc - val_acc:.2f}%\n")
        f.write(f"  Validation-Test Gap:  {val_acc - test_acc:.2f}%\n\n")
        
        f.write("CONFUSION MATRIX (Validation):\n")
        f.write(f"  [[{cm[0,0]:4d}  {cm[0,1]:4d}]\n")
        f.write(f"   [{cm[1,0]:4d}  {cm[1,1]:4d}]]\n\n")
        
        f.write("DIAGNOSIS:\n")
        if train_acc > 98 and (train_acc - val_acc) > 15:
            f.write("  ❌ SEVERE OVERFITTING DETECTED\n")
            f.write("  Model has likely memorized training data\n")
        elif train_acc > 95 and (train_acc - val_acc) > 10:
            f.write("  ⚠️  MODERATE OVERFITTING DETECTED\n")
            f.write("  Model generalization needs improvement\n")
        elif (train_acc - val_acc) < 5 and (val_acc - test_acc) < 5:
            f.write("  ✅ GOOD GENERALIZATION\n")
            f.write("  Model performs consistently across datasets\n")
        else:
            f.write("  🔍 CHECK FOR DATA LEAKAGE\n")
            f.write("  Unusual pattern - verify data splits\n")

if __name__ == "__main__":
    run_diagnostics()