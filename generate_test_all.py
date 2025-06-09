import os
import shutil
import random
from pathlib import Path
from collections import Counter
from collections import defaultdict

# Example usage
input_directory = "/home/ar/FLOCKD/NN_Classification/cifar100/test"
ref_directory = "/home/ar/FLOCKD/NN_Classification/train_1-100_5000-15_cifar100_similar_mod_range_2"
output_directory = "/home/ar/FLOCKD/NN_Classification/test_1-100_5000-15_cifar100_similar_mod_range_2"

def copy_directory_structure(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}")
        return
    
    for dirpath, dirnames, _ in os.walk(input_dir):
        for dirname in dirnames:
            src_path = Path(dirpath) / dirname
            dest_path = output_dir / src_path.relative_to(input_dir)
            dest_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dest_path}")

def list_and_count_subfolders(directory):
    directory = Path(directory)
    if not directory.is_dir():
        print(f"Input path is not a directory: {directory}")
        return [], [], defaultdict(list)
        
    class_list = []
    path_dict = defaultdict(list)
    for dirs in os.listdir(directory):
        dir_path = os.path.join(directory, dirs)
        if os.path.isdir(dir_path):
            for dir in os.listdir(dir_path):
                subdir_path = os.path.join(dir_path, dir)
                if os.path.isdir(subdir_path):
                    class_list.append(dir)
                    path_dict[dir].append(Path(directory) / dirs / dir)
            
    subfolders = Counter(class_list)
    distinct_names = list(subfolders.keys())
    counts = list(subfolders.values())
    return distinct_names, counts, path_dict

def distribute_images(input_dir, output_dir, images_per_class=50, seed=None):
    """
    Randomly selects 'images_per_class' images per class from input_dir 
    and copies them to ALL matching class folders in output_dir.
    """
    if seed is not None:
        random.seed(seed)  # For reproducibility
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}")
        return
    
    # Get all classes in output_dir and their paths
    _, _, path_dict = list_and_count_subfolders(output_dir)
    
    for class_name, dest_paths in path_dict.items():
        src_class_dir = input_dir / class_name
        
        if not src_class_dir.exists() or not src_class_dir.is_dir():
            print(f"Skipping missing class or non-directory: {class_name}")
            continue
        
        # Get all images (supports multiple extensions)
        images = list(src_class_dir.glob("*.png")) + list(src_class_dir.glob("*.jpg"))
        
        if len(images) < images_per_class:
            print(f"Skipping {class_name}: Not enough images ({len(images)} < {images_per_class})")
            continue
        
        # Randomly select images
        selected_images = random.sample(images, images_per_class)
        
        # Copy to ALL destination paths for this class
        for dest_path in dest_paths:
            dest_path.mkdir(parents=True, exist_ok=True)
            for img in selected_images:
                if img.is_file():  # Only copy if it's a file
                    shutil.copy2(img, dest_path / img.name)
            
            print(f"Copied {images_per_class} images to {dest_path}")

    print("Distribution complete!")

# Example Usage
copy_directory_structure(ref_directory, output_directory)

distribute_images(input_directory,
                  output_directory, 
                  images_per_class=70, 
                  seed=42)  # Optional: for reproducible randomness
                  