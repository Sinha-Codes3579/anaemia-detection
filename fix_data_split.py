# fix_data_split.py
import os
import re
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split
from config.config import Config

def extract_patient_id(filename):
    match = re.search(r'(?:FN|Fin|fn)-(\d+)', filename, re.IGNORECASE)
    return 'FN-' + match.group(1).zfill(3) if match else None

def clear_data_fixed(target_root):
    """Wipe and recreate all split/class folders"""
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
        print(f"🗑️  Cleared old data_fixed/")
    for split in ['train', 'val', 'test']:
        for cls in ['anaemic', 'non_anaemic']:
            os.makedirs(os.path.join(target_root, split, cls), exist_ok=True)
    print(f"📁 Recreated data_fixed/ folder structure")

def populate_data_fixed():
    config = Config()
    target_root = os.path.join(config.PROJECT_ROOT, 'data_fixed')

    # ── Step 1: Wipe and recreate ──────────────────────────────────────────────
    clear_data_fixed(target_root)

    # ── Step 2: Source folders ─────────────────────────────────────────────────
    source_classes = {
        'anaemic':     os.path.join(config.PROJECT_ROOT, 'data', 'anaemic'),
        'non_anaemic': os.path.join(config.PROJECT_ROOT, 'data', 'non anaemic'),
    }

    # Verify sources exist
    for cls, src in source_classes.items():
        if not os.path.exists(src):
            print(f"❌ Source not found: {src}")
            return
        files = [f for f in os.listdir(src) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        print(f"✅ {cls}: found {len(files)} images in {src}")

    # ── Step 3: Group by patient ID ────────────────────────────────────────────
    unknown_count = 0
    patient_files = {}   # { (cls, patient_id): [filepath, ...] }

    for cls_key, src_dir in source_classes.items():
        for filename in os.listdir(src_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            pid = extract_patient_id(filename)
            if pid is None:
                pid = f'UNK-{unknown_count:04d}'
                unknown_count += 1
            key = (cls_key, pid)
            if key not in patient_files:
                patient_files[key] = []
            patient_files[key].append(os.path.join(src_dir, filename))

    # Separate by class
    anaemic_patients     = {k[1]: v for k, v in patient_files.items() if k[0] == 'anaemic'}
    non_anaemic_patients = {k[1]: v for k, v in patient_files.items() if k[0] == 'non_anaemic'}

    print(f"\n📊 Grouped patients:")
    print(f"   anaemic     : {len(anaemic_patients)} patients, {sum(len(v) for v in anaemic_patients.values())} images")
    print(f"   non_anaemic : {len(non_anaemic_patients)} patients, {sum(len(v) for v in non_anaemic_patients.values())} images")

    # ── Step 4: Split by patient ID ────────────────────────────────────────────
    def split_patients(patient_dict, cls_name):
        pids = list(patient_dict.keys())
        train_p, temp_p = train_test_split(pids, test_size=0.30, random_state=42)
        val_p,  test_p  = train_test_split(temp_p, test_size=0.50, random_state=42)

        print(f"\n   {cls_name} split → train:{len(train_p)} val:{len(val_p)} test:{len(test_p)} patients")

        counts = {}
        for split_name, pid_list in [('train', train_p), ('val', val_p), ('test', test_p)]:
            n = 0
            for pid in pid_list:
                for src_path in patient_dict[pid]:
                    dst = os.path.join(target_root, split_name, cls_name, os.path.basename(src_path))
                    shutil.copy2(src_path, dst)
                    n += 1
            counts[split_name] = n
            print(f"   {split_name:5s}/{cls_name}: {n} images copied")

        # Verify no overlap
        assert not set(train_p) & set(val_p),  "❌ train/val patient overlap!"
        assert not set(train_p) & set(test_p), "❌ train/test patient overlap!"
        assert not set(val_p)  & set(test_p),  "❌ val/test patient overlap!"
        print(f"   ✅ Zero patient overlap confirmed")

    split_patients(anaemic_patients,     'anaemic')
    split_patients(non_anaemic_patients, 'non_anaemic')

    # ── Step 5: Final verification ─────────────────────────────────────────────
    print("\n" + "="*50)
    print("FINAL IMAGE COUNTS")
    print("="*50)
    total = 0
    for split in ['train', 'val', 'test']:
        for cls in ['anaemic', 'non_anaemic']:
            path = os.path.join(target_root, split, cls)
            n = len(os.listdir(path))
            total += n
            print(f"  {split:5s}/{cls}: {n}")
    print(f"  {'TOTAL':>17}: {total}")
    print(f"\n✅ data_fixed/ is ready. Now retrain the model.")

if __name__ == '__main__':
    populate_data_fixed()