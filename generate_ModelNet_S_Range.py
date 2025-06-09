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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pandas as pd
import json

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
    class_names = list(class_embeddings.keys())
    vectors = np.stack([class_embeddings[c] for c in class_names])
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed)
    cluster_ids = kmeans.fit_predict(vectors)

    clustered_classes = defaultdict(list)
    for class_name, cluster_id in zip(class_names, cluster_ids):
        clustered_classes[cluster_id].append(class_name)

    return clustered_classes

def generate_flexible_similar_subsets_with_visualization(input_dir, 
                                                         output_dir, 
                                                         save_path, 
                                                         clustered_classes, 
                                                         class_embeddings, 
                                                         num_subsets=5000, 
                                                         subset_size=15, 
                                                         images_per_class=400, 
                                                         top_k_clusters=3, 
                                                         copy_method='copy', 
                                                         random_seed=42, 
                                                         similarity_range=(0.6, 0.85)):
    def compute_cluster_centroids():
        return {cid: np.mean([class_embeddings[c] for c in clist], axis=0) for cid, clist in clustered_classes.items()}

    def find_similar_clusters(cluster_centroids, top_k=3):
        similarity_map = {}
        for i, ci in cluster_centroids.items():
            sims = []
            for j, cj in cluster_centroids.items():
                if i == j: continue
                sim = 1 - cosine(ci, cj)
                sims.append((j, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            similarity_map[i] = [cid for cid, _ in sims[:top_k]]

        return similarity_map

    def compute_subset_similarity(class_list):
        sims = []
        for i in range(len(class_list)):
            for j in range(i + 1, len(class_list)):
                emb1 = class_embeddings[class_list[i]]
                emb2 = class_embeddings[class_list[j]]
                sims.append(1 - cosine(emb1, emb2))
        return np.mean(sims)

    def plot_similarity_histogram(similarities, save_path):
        plt.figure(figsize=(8, 5))
        sns.histplot(similarities, bins=30, kde=True, color="skyblue")
        plt.title("Similarity in Flexible Similar-Class Subsets")
        plt.xlabel("Average Cosine Similarity")
        plt.ylabel("Number of Subsets")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "subset_similarity_distribution_heatmap.png"))  # Save heatmap
        plt.close()

    # Setup
    random.seed(random_seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_centroids = compute_cluster_centroids()
    similar_clusters_map = find_similar_clusters(cluster_centroids, top_k=top_k_clusters)

    subset_class_lists = []
    subset_similarities = []

    attempts = 0
    while len(subset_class_lists) < num_subsets and attempts < num_subsets * 5:
        attempts += 1
        base_cluster = random.choice(list(clustered_classes.keys()))
        cluster_pool = [base_cluster] + similar_clusters_map[base_cluster]  #[np.int32(1), np.int32(3), np.int32(10), np.int32(11)]

        available_classes = []
        for cid in cluster_pool:
            available_classes.extend(clustered_classes[cid])
        print(f"Available classes: {available_classes}")
        input()
        if len(available_classes) < subset_size:
            continue

        chosen_classes = random.sample(available_classes, subset_size)
        avg_sim = compute_subset_similarity(chosen_classes)

        if similarity_range[0] <= avg_sim <= similarity_range[1]:
            subset_idx = len(subset_class_lists)
            subset_output_dir = output_dir / f"subset_{subset_idx + 1}"
            subset_output_dir.mkdir(parents=True, exist_ok=True)

            for class_name in chosen_classes:
                src_class_dir = input_dir / class_name
                dst_class_dir = subset_output_dir / class_name
                dst_class_dir.mkdir(parents=True, exist_ok=True)

                image_files = list(src_class_dir.glob("*"))
                if len(image_files) < images_per_class:
                    raise ValueError(f"Not enough images in class {class_name}")
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
                        raise ValueError("Invalid copy_method")

            subset_class_lists.append(chosen_classes)
            subset_similarities.append(avg_sim)
            print(f"✅ Created subset {subset_idx + 1} with similarity {avg_sim:.4f}")

    print("🔍 Finished generating subsets.")
    plot_similarity_histogram(subset_similarities, save_path)

    sim_df = pd.DataFrame({"subset_id": list(range(1, len(subset_similarities) + 1)), 
                           "avg_cosine_similarity": subset_similarities})
    sim_df.to_csv(output_dir / "subset_similarity_scores.csv", index=False)
    print(f"📁 Saved similarity scores to: {output_dir / 'subset_similarity_scores.csv'}")

    subset_class_map = {f"subset_{i + 1}": subset_class_lists[i] for i in range(len(subset_class_lists))}
    with open(output_dir / "subset_class_mappings.json", "w") as f:
        json.dump(subset_class_map, f, indent=2)
    print(f"📁 Saved class mappings to: {output_dir / 'subset_class_mappings.json'}")

# -------------------- Example Usage --------------------
if __name__ == "__main__":
    input_dir = "/home/ar/FLOCKD/NN_Classification/cifar100/train"
    output_dir = "/home/ar/FLOCKD/NN_Classification/train_1-100_5000-15_cifar100_similar_mod_range_2"
    save_path = "/home/ar/FLOCKD/NN_Classification/results/data_split_train_1-100_5000-15_similar_mod_range_2"

    class_embeddings = get_class_representations(input_dir, max_images_per_class=90)
    clusters = cluster_classes(class_embeddings, num_clusters=15)
    generate_flexible_similar_subsets_with_visualization(input_dir=input_dir,
                                                         output_dir=output_dir, 
                                                         save_path=save_path, 
                                                         clustered_classes=clusters, 
                                                         class_embeddings=class_embeddings, 
                                                         num_subsets=5000, 
                                                         subset_size=15, 
                                                         images_per_class=400, 
                                                         top_k_clusters=3, 
                                                         copy_method='copy', 
                                                         random_seed=42, 
                                                         similarity_range=(0.8, 0.95))
# This script generates flexible similar-class subsets from a dataset, ensuring that the average cosine 
# similarity between classes in each subset falls within a specified range. 
# It also visualizes the distribution of similarities across all generated subsets.