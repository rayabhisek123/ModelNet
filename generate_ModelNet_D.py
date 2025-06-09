import os
import shutil
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
import torchvision.models as models

# -------------------- CONFIG --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = models.resnet50(pretrained=True).to(DEVICE).eval()
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# ------------------------------------------------

def get_class_representations(dataset_dir, max_images_per_class=50):
    """
    Extracts average image embeddings per class using ResNet-50.
    """
    dataset_dir = Path(dataset_dir)
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    class_dirs.sort()

    class_embeddings = {}
    for class_dir in class_dirs:
        embeddings = []
        image_files = list(class_dir.glob("*"))
        for img_path in image_files[:max_images_per_class]:
            try:
                image = Image.open(img_path).convert("RGB")
                image = TRANSFORM(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    features = MODEL(image).squeeze().cpu().numpy()
                embeddings.append(features)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
        if embeddings:
            class_embeddings[class_dir.name] = np.mean(embeddings, axis=0)
        print(f"Processed class {class_dir.name}: {len(embeddings)} embeddings")
    return class_embeddings

def cluster_classes(class_embeddings, num_clusters=15, random_seed=42):
    """
    Clusters the class embeddings into groups of similar classes.
    """
    class_names = list(class_embeddings.keys())
    vectors = np.stack([class_embeddings[c] for c in class_names])
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed)
    cluster_ids = kmeans.fit_predict(vectors)

    clustered_classes = defaultdict(list)
    for class_name, cluster_id in zip(class_names, cluster_ids):
        clustered_classes[cluster_id].append(class_name)

    return clustered_classes

def generate_dissimilar_subsets(input_dir, output_dir, clustered_classes,
                                num_subsets=5000, images_per_class=400, 
                                copy_method='copy', random_seed=42):
    """
    Creates subsets of dissimilar classes (1 class per cluster).
    """
    random.seed(random_seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subset_idx in range(num_subsets):
        subset_output_dir = output_dir / f"subset_{subset_idx + 1}"
        subset_output_dir.mkdir(parents=True, exist_ok=True)

        chosen_classes = []
        for cluster in clustered_classes.values():
            chosen_class = random.choice(cluster)
            chosen_classes.append(chosen_class)

        for class_name in chosen_classes:
            src_class_dir = input_dir / class_name
            dst_class_dir = subset_output_dir / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            image_files = list(src_class_dir.glob("*"))
            if len(image_files) < images_per_class:
                raise ValueError(f"Not enough images in class {class_name} for sampling.")
            sampled_images = random.sample(image_files, images_per_class)

            for img_file in sampled_images:
                dest_path = dst_class_dir / img_file.name
                if copy_method == 'copy':
                    shutil.copy2(img_file, dest_path)
                elif copy_method == 'move':
                    shutil.move(img_file, dest_path)
                elif copy_method == 'symlink':
                    dest_path.symlink_to(img_file.resolve())
                else:
                    raise ValueError("Invalid copy_method. Use 'copy', 'move', or 'symlink'.")

        print(f"Created subset {subset_idx + 1} with {len(chosen_classes)} dissimilar classes.")

    print("All subsets generated successfully.")

# -------------------- Example Usage --------------------
# Uncomment and set the correct paths before running
input_dir = "/home/ar/FLOCKD/NN_Classification/cifar100/train"
output_dir = "/home/ar/FLOCKD/NN_Classification/train_1-100_5000-15_cifar100_KNN"
embeddings = get_class_representations(input_dir, max_images_per_class=75)
clusters = cluster_classes(embeddings, num_clusters=15)
generate_dissimilar_subsets(input_dir, output_dir, clusters, num_subsets=5000, images_per_class=400, copy_method='copy')
