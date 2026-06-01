import os

def check_directory_structure():
    base_path = "/content/drive/MyDrive/anaemiadetect"
    
    print("=== Checking Directory Structure ===")
    
    def list_all_files(path, indent=0):
        prefix = "  " * indent
        try:
            if os.path.exists(path):
                items = os.listdir(path)
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        print(f"{prefix}📁 {item}/")
                        list_all_files(item_path, indent + 1)
                    else:
                        # Only show image files
                        if item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            print(f"{prefix}🖼️  {item}")
            else:
                print(f"{prefix}❌ Path does not exist: {path}")
        except PermissionError:
            print(f"{prefix}🚫 Permission denied: {path}")
    
    # Check the main project directory
    print(f"\nProject directory: {base_path}")
    list_all_files(base_path)
    
    # Specifically check the raw images path you're using
    raw_path = "/content/drive/MyDrive/anaemiadetect/Anaemic/data/raw/images"
    print(f"\nSpecifically checking: {raw_path}")
    list_all_files(raw_path)

if __name__ == "__main__":
    check_directory_structure()