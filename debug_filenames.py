import os

data_dir = "/content/drive/MyDrive/anaemiadetect/Anaemia/data"

print("Sample filenames from anaemic folder:")
anaemic_dir = os.path.join(data_dir, "anaemic")
if os.path.exists(anaemic_dir):
    files = os.listdir(anaemic_dir)
    for f in files[:10]:
        print(f"  {f}")

print("\nSample filenames from non anaemic folder:")
non_anaemic_dir = os.path.join(data_dir, "non anaemic")
if os.path.exists(non_anaemic_dir):
    files = os.listdir(non_anaemic_dir)
    for f in files[:10]:
        print(f"  {f}")