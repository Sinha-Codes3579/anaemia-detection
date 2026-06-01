# check_data.py
import os

def check_data_structure():
    base_path = "/content/drive/MyDrive/anaemiadetect/Anaemia/data"
    
    print("=== Checking Data Structure ===")
    print(f"Base path: {base_path}")
    
    if not os.path.exists(base_path):
        print("❌ Data directory doesn't exist!")
        return
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            image_files = [f for f in os.listdir(item_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"📁 {item}: {len(image_files)} images")
            
            # Show first few files
            for img in image_files[:3]:
                print(f"   🖼️  {img}")
            if len(image_files) > 3:
                print(f"   ... and {len(image_files) - 3} more")
        else:
            print(f"📄 {item}")

if __name__ == "__main__":
    check_data_structure()