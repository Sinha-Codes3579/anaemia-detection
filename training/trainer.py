import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class AnemiaTrainer:
    def __init__(self, config, model_type='simple'):
        self.config = config
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train(self, use_nail_detection=False):
      from utils.data_loader import DataManager
      from models.anaemia_model import create_model


      data_manager = DataManager(self.config)
      data_manager.discover_data()
      train_loader, val_loader, test_loader = data_manager.get_data_loaders()

      model = create_model(self.model_type, num_classes=self.config.NUM_CLASSES)
      model.to(self.device)

      if hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(self.config.LEARNING_RATE)
        optimizer = Adam(param_groups, weight_decay=self.config.WEIGHT_DECAY)
        print(f"Using differential LR: backbone={self.config.LEARNING_RATE/10:.6f}, classifier={self.config.LEARNING_RATE:.6f}")
      else:
        optimizer = Adam(model.parameters(), lr=self.config.LEARNING_RATE,
                     weight_decay=self.config.WEIGHT_DECAY)

      # ── Weighted loss for class imbalance ──────────────────────────────────
      n_anaemic     = data_manager.train_labels.count(0)
      n_non_anaemic = data_manager.train_labels.count(1)
      total         = n_anaemic + n_non_anaemic
      w0 = total / (2 * n_anaemic)
      w1 = total / (2 * n_non_anaemic)
      class_weights = torch.tensor([w0, w1], dtype=torch.float).to(self.device)
      criterion = nn.CrossEntropyLoss(weight=class_weights)
      print(f"Class weights → anaemic: {w0:.3f}, non_anaemic: {w1:.3f}")

      optimizer = Adam(model.parameters(),
                 lr=self.config.LEARNING_RATE,
                 weight_decay=self.config.WEIGHT_DECAY)
      scheduler = ReduceLROnPlateau(optimizer, mode='max',  # ← track accuracy
                                   patience=5, factor=0.5)

      best_val_acc = 0.0          # ← save on accuracy, not loss
      patience_counter = 0

      print(f"Starting training with {self.model_type} model...")
      print(f"Training samples:   {len(data_manager.train_paths)}")
      print(f"Validation samples: {len(data_manager.val_paths)}")
      print(f"Test samples:       {len(data_manager.test_paths)}")

      for epoch in range(self.config.NUM_EPOCHS):

        # ── Train phase ────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader,
                    desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]')
        for images, labels in pbar:
          images, labels = images.to(self.device), labels.to(self.device)

          optimizer.zero_grad()
          outputs = model(images)
          loss    = criterion(outputs, labels)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()

          train_loss    += loss.item()
          _, predicted   = torch.max(outputs.data, 1)
          train_total   += labels.size(0)
          train_correct += (predicted == labels).sum().item()
          pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc':  f'{100*train_correct/train_total:.2f}%'
          })

        # ── Validation phase ───────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
          for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        # ── Metrics ────────────────────────────────────────────────────
        train_acc      = 100 * train_correct / train_total
        val_acc        = 100 * val_correct   / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        self.train_losses.append(avg_train_loss)
        self.val_losses.append(avg_val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

        scheduler.step(val_acc)

        print(f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val   Loss: {avg_val_loss:.4f},   Val   Acc: {val_acc:.2f}%')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # ── Checkpoint ─────────────────────────────────────────────────
        if val_acc > best_val_acc:
          best_val_acc     = val_acc
          patience_counter = 0
          save_path = self.config.MODEL_SAVE_PATH.replace(
              '.pth', f'_{self.model_type}.pth')
          torch.save(model.state_dict(), save_path)
          print(f'  → Best val acc {best_val_acc:.2f}% — model saved!')
        else:
          patience_counter += 1
          print(f'  → Early stopping: {patience_counter}/'
                f'{self.config.EARLY_STOPPING_PATIENCE}')

        if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
          print("Early stopping triggered!")
          break

      # ── Post-training ──────────────────────────────────────────────────
      self.plot_training_history()

      model.eval()
      test_correct, test_total = 0, 0
      with torch.no_grad():
        for images, labels in test_loader:
          images, labels = images.to(self.device), labels.to(self.device)
          outputs      = model(images)
          _, predicted = torch.max(outputs.data, 1)
          test_total   += labels.size(0)
          test_correct += (predicted == labels).sum().item()

      test_acc = 100 * test_correct / test_total
      print(f"Final Test Accuracy: {test_acc:.2f}%")
      print(f"Best Val  Accuracy:  {best_val_acc:.2f}%")
      print("Training completed!")
      return model
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.RESULTS_DIR, f'training_history_{self.model_type}.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Training history plot saved to: {plot_path}")