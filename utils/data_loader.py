import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class YCbCrEnhance(A.ImageOnlyTransform):
    """Convert to YCbCr, enhance Cb/Cr channels, convert back — highlights nail pallor"""
    def apply(self, img, **params):
        ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        # Enhance chrominance channels (Cb, Cr carry color/pallor info)
        ycbcr = ycbcr.astype(np.float32)
        ycbcr[:,:,1] = np.clip(ycbcr[:,:,1] * 1.3, 0, 255)  # Cr
        ycbcr[:,:,2] = np.clip(ycbcr[:,:,2] * 1.3, 0, 255)  # Cb
        ycbcr = ycbcr.astype(np.uint8)
        return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
    
    def get_transform_init_args_names(self):
        return ()
class NailDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise ValueError(f"Could not load image: {self.image_paths[idx]}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.labels[idx]
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a dummy image and label
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, 0

class DataManager:
    def __init__(self, config):
        self.config = config
        self.train_paths = []
        self.train_labels = []
        self.val_paths = []
        self.val_labels = []
        self.test_paths = []
        self.test_labels = []
        
    def discover_data(self):
      """Read from pre-split data_fixed/ structure"""
      print("Loading data from pre-split data_fixed/ structure...")

      split_map = {
          'train': (self.train_paths, self.train_labels),
          'val':   (self.val_paths,   self.val_labels),
          'test':  (self.test_paths,  self.test_labels),
      }
      class_map = {'anaemic': 0, 'non_anaemic': 1}

      for split_name, (paths_list, labels_list) in split_map.items():
          for cls_name, cls_idx in class_map.items():
              folder = os.path.join(self.config.DATA_DIR, split_name, cls_name)
              if not os.path.exists(folder):
                  print(f"⚠️  Missing folder: {folder}")
                  continue
              files = [
                  f for f in os.listdir(folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))
              ]
              for f in files:
                  paths_list.append(os.path.join(folder, f))
                  labels_list.append(cls_idx)

      print(f"✅ Train : {len(self.train_paths)} images")
      print(f"✅ Val   : {len(self.val_paths)} images")
      print(f"✅ Test  : {len(self.test_paths)} images")

      assert len(self.train_paths) > 0, "❌ No training images found!"
      assert len(self.val_paths)   > 0, "❌ No validation images found!"
      assert len(self.test_paths)  > 0, "❌ No test images found!"

      return self
    
    def get_transforms(self):
      train_transform = A.Compose([
        YCbCrEnhance(p=1.0),                    # ← ADD: color enhancement first
        A.Resize(*self.config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
        # REMOVE: A.ToColorJitter(p=0.0)        ← delete this line
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
      ])

      val_transform = A.Compose([
        YCbCrEnhance(p=1.0),                    # ← ADD: apply to val too
        A.Resize(*self.config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
      ])

      return train_transform, val_transform
    
    def get_data_loaders(self):
        """Get data loaders for training and validation"""
        train_transform, val_transform = self.get_transforms()
        
        train_dataset = NailDataset(self.train_paths, self.train_labels, train_transform)
        val_dataset = NailDataset(self.val_paths, self.val_labels, val_transform)
        test_dataset = NailDataset(self.test_paths, self.test_labels, val_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=self.config.NUM_WORKERS
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.config.NUM_WORKERS
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.config.NUM_WORKERS
        )
        
        return train_loader, val_loader, test_loader