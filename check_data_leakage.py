import os
import re
from config.config import Config
from utils.data_loader import DataManager

config = Config()
dm = DataManager(config)
dm.discover_data()

def extract_patient_id(path):
    """Extract patient ID from your filename pattern (e.g., FN-001)"""
    # ADAPT THIS PATTERN TO YOUR FILENAMES
    match = re.search(r'(FN-\d+)', os.path.basename(path))
    return match.group(0) if match else "unknown"

def check_leakage(split1_paths, split1_name, split2_paths, split2_name):
    """Check for duplicates between two splits"""
    set1 = set(split1_paths)
    set2 = set(split2_paths)
    
    # Check exact file duplicates
    file_overlap = set1.intersection(set2)
    
    # Check patient ID duplicates
    patient1 = set([extract_patient_id(p) for p in split1_paths])
    patient2 = set([extract_patient_id(p) for p in split2_paths])
    patient_overlap = patient1.intersection(patient2)
    
    print(f"\n {split1_name} vs {split2_name}:")
    print(f"   File duplicates: {len(file_overlap)}")
    if file_overlap:
        print(f"   Example duplicates: {list(file_overlap)[:3]}")
    print(f"   Patient ID overlap: {len(patient_overlap)}")
    if patient_overlap:
        print(f"   Shared patients: {list(patient_overlap)[:5]}")
    
    return len(file_overlap) > 0 or len(patient_overlap) > 0

print("="*60)
print("DATA LEAKAGE INVESTIGATION")
print("="*60)

leak_found = False
leak_found |= check_leakage(dm.train_paths, "TRAIN", dm.val_paths, "VALIDATION")
leak_found |= check_leakage(dm.train_paths, "TRAIN", dm.test_paths, "TEST")
leak_found |= check_leakage(dm.val_paths, "VALIDATION", dm.test_paths, "TEST")

if leak_found:
    print("\n Critical: Data leakage detected!")
    print("   This invalidates your accuracy metrics.")
else:
    print("\n No file or patient ID leakage found.")