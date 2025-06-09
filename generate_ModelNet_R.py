###############------------Split(Not Shuffle) Dataset------------################
'''
import os
import shutil
import random
from pathlib import Path

def split_dataset(input_dir, output_dir, num_subsets=25, classes_per_subset=4):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    class_dirs.sort()  # Ensure consistent ordering
    
    if len(class_dirs) < num_subsets * classes_per_subset:
        raise ValueError("Not enough classes to split into the required subsets.")
    
    # Shuffle classes for randomness
    random.shuffle(class_dirs)
    
    for i in range(num_subsets):
        subset_classes = class_dirs[i * classes_per_subset: (i + 1) * classes_per_subset]
        subset_output_dir = output_dir / f'subset_{i+1}'
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        
        for class_dir in subset_classes:
            target_class_dir = subset_output_dir / class_dir.name
            shutil.copytree(class_dir, target_class_dir)
            print(f"Copied {class_dir} -> {target_class_dir}")
    
    print("Dataset splitting completed!")

# Example usage
input_directory = "/home/ar/CAD/Resnet50_CIFAR100/CIFAR-100-dataset-main/train"
output_directory = "/home/ar/CAD/Resnet50_CIFAR100/25_4_CIFAR-100-dataset-main_shuffle"
split_dataset(input_directory, output_directory)
'''

###############------------Split(Not Shuffle) Dataset------------################




###############------------Split(Shuffle) Dataset------------################
'''
import os
import shutil
import random
from pathlib import Path

def split_dataset_with_replacement(input_dir, output_dir, num_subsets=1000, classes_per_subset=10):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    class_dirs.sort()  # Ensure consistent ordering
    
    if len(class_dirs) < classes_per_subset:
        raise ValueError("Not enough classes for the required classes_per_subset.")
    
    for i in range(num_subsets):
        # Randomly sample classes with replacement for each subset
        subset_classes = random.sample(class_dirs, k=classes_per_subset)
        subset_output_dir = output_dir / f'subset_{i+1}'
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        
        for class_dir in subset_classes:
            target_class_dir = subset_output_dir / class_dir.name
            shutil.copytree(class_dir, target_class_dir, dirs_exist_ok=True)
            print(f"Copied {class_dir} -> {target_class_dir}")
    
    print(f"Dataset splitting completed! Created {num_subsets} subsets.")

# Example usage
input_directory = "/home/ar/CAD/Resnet50_CIFAR100/CIFAR-100-dataset-main/train"
output_directory = "/home/ar/CAD/Resnet50_CIFAR100/25_4_CIFAR-100-dataset-main_shuffle"
split_dataset_with_replacement(input_directory, output_directory)
'''
###############------------Split(Shuffle) Dataset------------################



###############------------Split(Shuffle) Dataset------------################
'''
import os
import shutil
import random
from pathlib import Path

def split_dataset_with_replacement(input_dir, output_dir, num_subsets=1000, classes_per_subset=10, random_seed=None):
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    class_dirs.sort()  # Ensure consistent ordering
    
    if len(class_dirs) < classes_per_subset:
        raise ValueError("Not enough classes for the required classes_per_subset.")
    
    for i in range(num_subsets):
        # Randomly sample classes with replacement for each subset
        subset_classes = random.sample(class_dirs, k=classes_per_subset)
        subset_output_dir = output_dir / f'subset_{i+1}'
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        
        for class_dir in subset_classes:
            target_class_dir = subset_output_dir / class_dir.name
            shutil.copytree(class_dir, target_class_dir, dirs_exist_ok=True)
            print(f"Copied {class_dir} -> {target_class_dir}")
    
    print(f"Dataset splitting completed! Created {num_subsets} subsets.")

# Example usage with random seed
input_directory = "/home/ar/CAD/Resnet50_CIFAR100/cifar100/train"
output_directory = "/home/ar/CAD/Resnet50_CIFAR100/data_train_1-100_5000-25_shuffle"
split_dataset_with_replacement(input_directory, 
                               output_directory,
                               num_subsets=5000, 
                               classes_per_subset=25, 
                               random_seed=42  # Fixed seed for reproducibility
                               )
'''
###############------------Split(Shuffle) Dataset------------################



###############------------Split(Shuffle) Dataset------------################
import os
import shutil
import random
from pathlib import Path

def split_dataset_with_replacement(input_dir, 
                                   output_dir, 
                                   num_subsets=1000, 
                                   classes_per_subset=10, 
                                   images_per_class=None, 
                                   random_seed=None,
                                   copy_method='copy'):  # 'copy', 'move', or 'symlink'
    """
    Split dataset into subsets with controlled random sampling of images.
    
    Args:
        input_dir: Input directory containing class folders
        output_dir: Output directory for subsets
        num_subsets: Number of subsets to create
        classes_per_subset: Number of classes per subset
        images_per_class: Number of images to sample per class (None for all)
        random_seed: Seed for reproducible randomness
        copy_method: 'copy', 'move', or 'symlink' for file handling
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    class_dirs.sort()  # Ensure consistent ordering
    
    if len(class_dirs) < classes_per_subset:
        raise ValueError("Not enough classes for the required classes_per_subset.")
    
    for i in range(num_subsets):
        # Randomly sample classes with replacement for each subset
        subset_classes = random.sample(class_dirs, k=classes_per_subset)
        subset_output_dir = output_dir / f'subset_{i+1}'
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        
        for class_dir in subset_classes:
            target_class_dir = subset_output_dir / class_dir.name
            target_class_dir.mkdir(exist_ok=True)
            
            # Get all images in the class directory
            image_files = [f for f in class_dir.iterdir() if f.is_file()]
            
            # Sample images if requested
            if images_per_class is not None:
                if len(image_files) < images_per_class:
                    raise ValueError(f"Class {class_dir.name} has only {len(image_files)} images, "f"but {images_per_class} requested.")
                image_files = random.sample(image_files, images_per_class)
            
            # Handle files according to specified method
            for img_file in image_files:
                dest_path = target_class_dir / img_file.name
                if copy_method == 'copy':
                    shutil.copy2(img_file, dest_path)  # Preserves metadata
                elif copy_method == 'move':
                    shutil.move(img_file, dest_path)
                elif copy_method == 'symlink':
                    dest_path.symlink_to(img_file.resolve())
                else:
                    raise ValueError("copy_method must be 'copy', 'move', or 'symlink'")
            
            print(f"Processed {len(image_files)} images from {class_dir} -> {target_class_dir}")
    
    print(f"Dataset splitting completed! Created {num_subsets} subsets.")

# Example usage with image sampling
input_directory = "/home/ar/FLOCKD/NN_Classification/cifar100/train"
output_directory = "/home/ar/FLOCKD/NN_Classification/train_1-100_5000-15_cifar100_shuffle"

split_dataset_with_replacement(input_directory, 
                               output_directory,
                               num_subsets=5000, 
                               classes_per_subset=15, 
                               images_per_class=400,  # Sample 400 images per class
                               random_seed=42,       # Fixed seed for reproducibility
                               copy_method='copy')    # Options: 'copy', 'move', 'symlink'
                               
###############------------Split(Shuffle) Dataset------------################
