# organize_data.py
import os
import shutil

def organize_data():
    """Organize the messy data into proper class directories"""
    
    # Use your actual project path
    project_root = "/content/drive/MyDrive/anaemiadetect/Anaemia"
    source_dir = "/content/drive/MyDrive/anaemiadetect"  # Where your images currently are
    target_dir = os.path.join(project_root, "organized_data")
    
    # Create class directories
    anaemic_dir = os.path.join(target_dir, "anaemic")
    non_anaemic_dir = os.path.join(target_dir, "non_anaemic")
    
    os.makedirs(anaemic_dir, exist_ok=True)
    os.makedirs(non_anaemic_dir, exist_ok=True)
    
    anaemic_count = 0
    non_anaemic_count = 0
    
    print("Starting data organization...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    # Organize files
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            source_path = os.path.join(source_dir, filename)
            
            # Determine class based on filename
            filename_lower = filename.lower()
            
            if 'non' in filename_lower or 'non-anrmic' in filename_lower or 'non-anemic' in filename_lower:
                target_path = os.path.join(non_anaemic_dir, filename)
                shutil.copy2(source_path, target_path)
                non_anaemic_count += 1
                print(f"✓ Copied {filename} to non_anaemic")
            elif 'anaemic' in filename_lower or 'anrmic' in filename_lower:
                target_path = os.path.join(anaemic_dir, filename)
                shutil.copy2(source_path, target_path)
                anaemic_count += 1
                print(f"✓ Copied {filename} to anaemic")
            else:
                # Default to non-anaemic for unknown files
                target_path = os.path.join(non_anaemic_dir, filename)
                shutil.copy2(source_path, target_path)
                non_anaemic_count += 1
                print(f"? Unknown class for {filename}, defaulted to non_anaemic")
    
    print(f"\nData organization completed!")
    print(f"Anaemic images: {anaemic_count}")
    print(f"Non-anaemic images: {non_anaemic_count}")
    print(f"Total images: {anaemic_count + non_anaemic_count}")

if __name__ == "__main__":
    organize_data()