import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score
sns.set(style="whitegrid")



def extract_class_embeddings(dataset_dir, model, transform, device, max_images_per_class=90, cache_file="class_embeddings.pkl"):
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    class_dirs = [d for d in Path(dataset_dir).iterdir() if d.is_dir()]
    class_embeddings = {}
    model.eval()

    for class_dir in tqdm(class_dirs, desc="Extracting class embeddings"):
        image_paths = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        image_paths = image_paths[:max_images_per_class]
        images = [transform(Image.open(p).convert('RGB')) for p in image_paths]
        if not images:
            continue
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            feats = model(batch)
        class_embeddings[class_dir.name] = feats.mean(dim=0).cpu().numpy()

    with open(cache_file, 'wb') as f:
        pickle.dump(class_embeddings, f)
    print(f"Saved embeddings to {cache_file}")
    return class_embeddings

def compute_average_subset_diversity(subsets_dir, class_embeddings):
    diversity_scores = []
    subsets_dir = Path(subsets_dir)
    subset_dirs = [d for d in subsets_dir.iterdir() if d.is_dir()]

    for subset in subset_dirs:
        class_dirs = [d for d in subset.iterdir() if d.is_dir()]
        try:
            vectors = [class_embeddings[d.name] for d in class_dirs if d.name in class_embeddings]
            if len(vectors) < 2:
                continue
            distances = cosine_distances(vectors)
            upper_tri = distances[np.triu_indices_from(distances, k=1)]
            diversity_scores.append(upper_tri.mean())
        except Exception as e:
            print(f"Failed on {subset.name}: {e}")
    return diversity_scores

def compute_class_occurrences(subsets_dir):
    counter = Counter()
    subsets_dir = Path(subsets_dir)
    for subset in subsets_dir.iterdir():
        if subset.is_dir():
            for cls in subset.iterdir():
                if cls.is_dir():
                    counter[cls.name] += 1
    return counter

def compute_jaccard_similarity(knn_dir, rand_dir):
    knn_subsets = [set(d.name for d in Path(sub).iterdir() if d.is_dir()) for sub in Path(knn_dir).iterdir() if Path(sub).is_dir()]
    rand_subsets = [set(d.name for d in Path(sub).iterdir() if d.is_dir()) for sub in Path(rand_dir).iterdir() if Path(sub).is_dir()]
    similarities = []
    for k_set, r_set in zip(knn_subsets, rand_subsets):
        intersection = len(k_set & r_set)
        union = len(k_set | r_set)
        if union == 0:
            continue
        similarities.append(intersection / union)
    return similarities

def compute_class_coverage(subsets_dir):
    all_classes = set()
    for subset in Path(subsets_dir).iterdir():
        if Path(subset).is_dir():
            all_classes.update([d.name for d in subset.iterdir() if d.is_dir()])
    return len(all_classes)

def compute_subset_redundancy(subsets_dir):
    subset_dirs = [d for d in Path(subsets_dir).iterdir() if d.is_dir()]
    class_sets = [set(cls.name for cls in subset.iterdir() if cls.is_dir()) for subset in subset_dirs]
    redundancy = []
    for i in range(len(class_sets)):
        for j in range(i + 1, len(class_sets)):
            overlap = len(class_sets[i] & class_sets[j])
            redundancy.append(overlap)
    return redundancy

def compute_intra_subset_variance(subsets_dir, class_embeddings):
    variances = []
    for subset in Path(subsets_dir).iterdir():
        if not subset.is_dir():
            continue
        vectors = [class_embeddings[d.name] for d in subset.iterdir() if d.is_dir() and d.name in class_embeddings]
        if len(vectors) < 2:
            continue
        vecs = np.stack(vectors)
        var = np.var(vecs, axis=0).mean()
        variances.append(var)
    return variances

def compute_feature_space_coverage(subsets_dir, class_embeddings):
    embeddings = []
    for subset in Path(subsets_dir).iterdir():
        if Path(subset).is_dir():
            for d in subset.iterdir():
                if d.is_dir() and d.name in class_embeddings:
                    embeddings.append(class_embeddings[d.name])
    if not embeddings:
        return 0.0
    matrix = np.stack(embeddings)
    cov = np.cov(matrix, rowvar=False)
    return np.trace(cov)







##########------------------Visualization------------------##########
##### 1. Class Embedding Diversity
def compare_distributions(dist1, dist2, dist3, title="", xlabel="", ylabel="Frequency", output_path=None):
    plt.figure(figsize=(12, 6))
    sns.histplot(dist1, color='blue', label='ModelNet-R', kde=True, stat="density", bins=30)
    sns.histplot(dist2, color='orange', label='ModelNet-D', kde=True, stat="density", bins=30)
    sns.histplot(dist3, color='cyan', label='ModelNet-S', kde=True, stat="density", bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    plt.close()


##### 2. Class Occurrence Histogram
def plot_class_occurrence_histogram(rand_counts, knn_d_counts, knn_s_counts, output_path=None):
    # Union of all classes
    all_classes = sorted(set(rand_counts) | set(knn_d_counts) | set(knn_s_counts))

    # Create a list of dictionaries for DataFrame construction
    data = []
    for cls in all_classes:
        data.append({'Class': cls, 'Count': rand_counts.get(cls, 0), 'Method': 'ModelNet-R'})
        data.append({'Class': cls, 'Count': knn_d_counts.get(cls, 0), 'Method': 'ModelNet-D'})
        data.append({'Class': cls, 'Count': knn_s_counts.get(cls, 0), 'Method': 'ModelNet-S'})

    df = pd.DataFrame(data)

    plt.figure(figsize=(18, 6))
    sns.barplot(data=df, x='Class', y='Count', hue='Method', palette='pastel')

    plt.xlabel('Class Index')
    plt.ylabel('Subset Count')
    plt.title('Class Occurrence Frequency Across Subsets')
    plt.xticks(rotation='vertical')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved histogram to {output_path}")
    plt.close()
'''
def plot_class_occurrence_histogram1(rand_counts, knn_d_counts, knn_s_counts, output_path=None):
    all_classes = sorted(set(rand_counts.keys()) | set(knn_d_counts.keys()) | set(knn_s_counts.keys()))
    rand_values = [rand_counts.get(cls, 0) for cls in all_classes]
    knn_d_values = [knn_d_counts.get(cls, 0) for cls in all_classes]
    knn_s_values = [knn_s_counts.get(cls, 0) for cls in all_classes]

    x = np.arange(len(all_classes))
    width = 0.25

    plt.figure(figsize=(18, 6))
    plt.bar(x - width, knn_d_values, width, label='KNN_D')
    plt.bar(x, rand_values, width, label='Random')
    plt.bar(x + width, knn_s_values, width, label='KNN_S')

    plt.xlabel('Class Index')
    plt.ylabel('Subset Count')
    plt.title('Class Occurrence Frequency Across Subsets')
    plt.xticks(ticks=x, labels=all_classes, rotation='vertical')
    plt.legend()
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved histogram to {output_path}")
    plt.close()
'''


##### 3. t-SNE Plot for embeddingds

# ------------------------------
# 1. Modified ResNet50
# ------------------------------
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # remove avgpool and fc
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x).squeeze()
        return features  # [batch_size, 2048]

# ------------------------------
# 2. Embedding Extraction
# ------------------------------
def extract_and_cache_embeddings(dir_path, model, transform, cache_path, batch_size=32, device="cuda"):
    """
    Extracts embeddings from images using the model and caches to disk if not already saved.
    Returns (embeddings, labels, class_names).
    """
    if Path(cache_path).exists():
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['embeddings'], data['labels'], data['class_names'].tolist()

    print(f"Extracting and caching embeddings for {dir_path}")
    model.eval()
    loader = DataLoader(datasets.ImageFolder(dir_path, transform=transform), 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=4)

    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Extracting from {dir_path.name}"):
            imgs = imgs.to(device)
            feats = model(imgs).cpu().numpy()
            all_embeddings.append(feats)
            all_labels.extend(labels.numpy())

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    class_names = loader.dataset.classes

    # Save to cache
    np.savez(cache_path, embeddings=embeddings, labels=labels, class_names=class_names)
    print(f"Saved embeddings to {cache_path}")

    return embeddings, labels, class_names


# ------------------------------
# 3. t-SNE Computation
# ------------------------------
def compute_tsne(embeddings):
    pca = PCA(n_components=75)
    pca_feats = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    return tsne.fit_transform(pca_feats)

# ------------------------------
# 4. Visualization
# ------------------------------
def plot_tsne_triple(tsne_data_list, label_list, class_names, titles, output_path):
    plt.figure(figsize=(24, 8))

    for i, (tsne_data, labels, title) in enumerate(zip(tsne_data_list, label_list, titles)):
        ax = plt.subplot(1, 3, i + 1)
        sns.scatterplot(x=tsne_data[:, 0],
                        y=tsne_data[:, 1],
                        hue=[class_names[l] for l in labels],
                        palette="tab20",
                        alpha=0.7,
                        s=100,
                        edgecolor='w',
                        linewidth=0.5,
                        ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend([], [], frameon=False)
        ax.grid(True, linestyle='--', alpha=0.2)

    plt.suptitle("ResNet50 + PCA + t-SNE on Dataset Variants", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

# ------------------------------
# 5. Main Pipeline
# ------------------------------


def run_tsne_for_variants_cached(rand_dir, knn_d_dir, knn_s_dir, output_path="tsne_variants.png", cache_root="cached_embeddings"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet50FeatureExtractor(num_classes=15).to(device)

    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    Path(cache_root).mkdir(exist_ok=True)

    rand_emb, rand_labels, class_names = extract_and_cache_embeddings(Path(rand_dir), model, transform, f"{cache_root}/rand_embeddings.npz", device=device)
    knn_d_emb, knn_d_labels, _ = extract_and_cache_embeddings(Path(knn_d_dir), model, transform, f"{cache_root}/knn_d_embeddings.npz", device=device)
    knn_s_emb, knn_s_labels, _ = extract_and_cache_embeddings(Path(knn_s_dir), model, transform, f"{cache_root}/knn_s_embeddings.npz", device=device)

    # Compute t-SNE
    rand_tsne = compute_tsne(rand_emb)
    knn_d_tsne = compute_tsne(knn_d_emb)
    knn_s_tsne = compute_tsne(knn_s_emb)

    # Plot
    plot_tsne_triple([rand_tsne, knn_d_tsne, knn_s_tsne], 
                     [rand_labels, knn_d_labels, knn_s_labels], 
                     class_names, 
                     ["ModelNet-R", "ModelNet-D", "ModelNet-S"], 
                     output_path)

# Example usage:
# run_tsne_for_variants("data/ModelNet-R", "data/ModelNet-D", "data/ModelNet-S")



def extract_embeddings_for_subset(subset_dir, class_embeddings):
    """
    Extract embeddings for all classes in a given subset directory.
    """
    subset_embeddings = []
    subset_labels = []
    
    subset_path = Path(subset_dir)
    class_dirs = [d for d in subset_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in class_embeddings:
            continue
        embedding = class_embeddings[class_name]
        subset_embeddings.append(embedding)
        subset_labels.append(class_name)
        
    return np.array(subset_embeddings), subset_labels


def plot_tsne_triple_subplots(rand_dir, knn_d_dir, knn_s_dir, class_embeddings, output_path="tsne_triple_plot.png"):
    """
    Generate a single figure with 3 PCA+t-SNE scatterplots:
    - One for KNN-D
    - One for KNN-S
    - One for Random
    """
    def collect_embeddings(subset_dir):
        all_embeddings, all_labels = [], []
        for subset in sorted(Path(subset_dir).iterdir()):
            if not subset.is_dir():
                continue
            embeddings, labels = extract_embeddings_for_subset(subset, class_embeddings)
            all_embeddings.append(embeddings)
            all_labels.extend(labels)
        return np.vstack(all_embeddings), all_labels

    def compute_tsne(embeddings):
        pca = PCA(n_components=75)
        pca_embeddings = pca.fit_transform(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        return tsne.fit_transform(pca_embeddings)

    # Collect and compute embeddings
    rand_emb, rand_labels = collect_embeddings(rand_dir)
    knn_d_emb, knn_d_labels = collect_embeddings(knn_d_dir)
    knn_s_emb, knn_s_labels = collect_embeddings(knn_s_dir)

    rand_tsne = compute_tsne(rand_emb)
    knn_d_tsne = compute_tsne(knn_d_emb)
    knn_s_tsne = compute_tsne(knn_s_emb)

    # Create side-by-side subplots
    plt.figure(figsize=(24, 8))

    def plot_subplot(ax, tsne_data, labels, title, palette):
        sns.scatterplot(x=tsne_data[:, 0], 
                        y=tsne_data[:, 1], 
                        hue=labels, 
                        palette=palette, 
                        alpha=0.7, 
                        s=100, 
                        edgecolor='w', 
                        linewidth=0.5, 
                        ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
        ax.legend([], [], frameon=False)
        ax.grid(True, linestyle='--', alpha=0.2)

    # Plot each in a separate subplot
    ax1 = plt.subplot(1, 3, 1)
    plot_subplot(ax1, rand_tsne, rand_labels, "ModelNet-R", palette="plasma")
    
    ax2 = plt.subplot(1, 3, 2)
    plot_subplot(ax2, knn_d_tsne, knn_d_labels, "ModelNet-D", palette="coolwarm")

    ax3 = plt.subplot(1, 3, 3)
    plot_subplot(ax3, knn_s_tsne, knn_s_labels, "ModelNet-S", palette="viridis")

    # Supertitle
    plt.suptitle("PCA + t-SNE Visualization of Dataset Types", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved triple PCA+t-SNE plot to {output_path}")
    plt.close()


'''
def plot_tsne_for_subsets_improved(rand_dir, knn_d_dir, knn_s_dir, class_embeddings, output_path="tsne_plot_improved.png"):
    """
    Improved PCA+t-SNE visualization: KNN-D vs KNN-S vs Random subsets with better styling.
    """
    def collect_embeddings(subset_dir, label_prefix):
        all_embeddings, all_labels = [], []
        for subset in sorted(Path(subset_dir).iterdir()):
            if not subset.is_dir():
                continue
            embeddings, labels = extract_embeddings_for_subset(subset, class_embeddings)
            all_embeddings.append(embeddings)
            all_labels.extend([f"{label_prefix}_{label}" for label in labels])
        return np.vstack(all_embeddings), all_labels

    # Collect from three types
    knn_d_embeddings, knn_d_labels = collect_embeddings(knn_d_dir, "KNN-D")
    knn_s_embeddings, knn_s_labels = collect_embeddings(knn_s_dir, "KNN-S")
    rand_embeddings, rand_labels = collect_embeddings(rand_dir, "Random")

    # Combine for PCA + t-SNE
    all_embeddings = np.vstack([knn_d_embeddings, knn_s_embeddings, rand_embeddings])
    all_labels = knn_d_labels + knn_s_labels + rand_labels
    group_labels = (
        ["KNN-D"] * len(knn_d_embeddings) +
        ["KNN-S"] * len(knn_s_embeddings) +
        ["Random"] * len(rand_embeddings)
    )

    # PCA before t-SNE
    pca = PCA(n_components=50)
    pca_embeddings = pca.fit_transform(all_embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(pca_embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], 
                    y=tsne_results[:, 1], 
                    hue=group_labels, 
                    palette=["blue", "green", "orange"], 
                    alpha=0.7, 
                    s=100, 
                    edgecolor="w", 
                    linewidth=0.5)

    plt.title("PCA + t-SNE Visualization of KNN-D, KNN-S, and Random Subsets", fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Subset Type", loc="best")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved improved PCA+t-SNE plot to {output_path}")
    plt.close()

def plot_pca_tsne_for_subsets(rand_dir, knn_d_dir, knn_s_dir, class_embeddings, output_path="tsne_pca_plot.png"):
    # Containers
    knn_d_embeddings, knn_s_embeddings, rand_embeddings = [], [], []
    knn_d_labels, knn_s_labels, rand_labels = [], [], []

    # Load KNN-D
    for subset_dir in sorted(Path(knn_d_dir).iterdir()):
        if not subset_dir.is_dir():
            continue
        embeddings, labels = extract_embeddings_for_subset(subset_dir, class_embeddings)
        knn_d_embeddings.append(embeddings)
        knn_d_labels.extend(["KNN-D_" + l for l in labels])

    # Load KNN-S
    for subset_dir in sorted(Path(knn_s_dir).iterdir()):
        if not subset_dir.is_dir():
            continue
        embeddings, labels = extract_embeddings_for_subset(subset_dir, class_embeddings)
        knn_s_embeddings.append(embeddings)
        knn_s_labels.extend(["KNN-S_" + l for l in labels])

    # Load Random
    for subset_dir in sorted(Path(rand_dir).iterdir()):
        if not subset_dir.is_dir():
            continue
        embeddings, labels = extract_embeddings_for_subset(subset_dir, class_embeddings)
        rand_embeddings.append(embeddings)
        rand_labels.extend(["Random_" + l for l in labels])

    # Stack
    knn_d_embeddings = np.vstack(knn_d_embeddings) if knn_d_embeddings else np.empty((0, 0))
    knn_s_embeddings = np.vstack(knn_s_embeddings) if knn_s_embeddings else np.empty((0, 0))
    rand_embeddings   = np.vstack(rand_embeddings) if rand_embeddings else np.empty((0, 0))
    
    all_embeddings = np.vstack([knn_d_embeddings, knn_s_embeddings, rand_embeddings])
    all_labels = knn_d_labels + knn_s_labels + rand_labels
    group_labels = (["KNN-D"] * len(knn_d_embeddings) +
                    ["KNN-S"] * len(knn_s_embeddings) +
                    ["Random"] * len(rand_embeddings))

    # Step 1: PCA to 50D
    pca = PCA(n_components=50)
    pca_embeddings = pca.fit_transform(all_embeddings)

    # Step 2: t-SNE to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(pca_embeddings)

    # Step 3: Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_embeddings[:, 0], 
                    y=tsne_embeddings[:, 1], 
                    hue=group_labels, 
                    palette=["blue", "green", "orange"], 
                    alpha=0.7, 
                    s=100)
    plt.title("PCA + t-SNE Visualization of KNN-D, KNN-S, and Random Subsets", fontsize=14)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(title="Subset Type")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved PCA+t-SNE plot to {output_path}")
    plt.close()
'''



##### 4. Jaccard Similarity Plot
def plot_jaccard_similarity_comparison(jaccard_scores_1, 
                                       jaccard_scores_2, 
                                       jaccard_scores_3, 
                                       labels=("ModelNet-D vs ModelNet-R", "ModelNet-S vs ModelNet-R", "ModelNet-D vs ModelNet-S"), 
                                       output_path="jaccard_similarity_comparison.png"):
    means = [np.mean(jaccard_scores_1), np.mean(jaccard_scores_2), np.mean(jaccard_scores_3)]

    # Create a simple DataFrame for seaborn
    df = pd.DataFrame({"Method": labels, "Jaccard Similarity": means})
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="Method", y="Jaccard Similarity", palette='pastel')
    plt.ylim(0.08, 0.09)
    plt.ylabel("Jaccard Similarity")
    plt.title("Average Jaccard Similarity Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Jaccard plot to {output_path}")
    plt.close()


##### 5. Subset Redundancy Plot
def plot_subset_redundancy(rand_subset_redundancy, knn_d_subset_redundancy, knn_s_subset_redundancy, output_path="subset_redundancy.png"):
    # Prepare data for seaborn
    data = ([("ModelNet-R", val) for val in rand_subset_redundancy] + 
            [("ModelNet-D", val) for val in knn_d_subset_redundancy] + 
            [("ModelNet-S", val) for val in knn_s_subset_redundancy])

    df = pd.DataFrame(data, columns=["Method", "Redundancy"])

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Method", y="Redundancy", palette=["orange", "blue", "green"])
    plt.ylabel("Class Overlap Between Subsets")
    plt.title("Subset Redundancy")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved Subset Redundancy plot to {output_path}")
    plt.close()

##### 6. Intra-Subset Variance Plot
def plot_intra_subset_variance(rand_intra_variance, knn_d_intra_variance, knn_s_intra_variance, output_path="intra_subset_variance.png"):
    # Combine data into a long-form DataFrame
    data = ([("ModelNet-R", val) for val in rand_intra_variance] + 
            [("ModelNet-D", val) for val in knn_d_intra_variance] + 
            [("ModelNet-S", val) for val in knn_s_intra_variance])
    df = pd.DataFrame(data, columns=["Method", "Intra-Subset Variance"])

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Method", y="Intra-Subset Variance", palette=["orange", "blue", "green"])
    plt.ylabel("Mean Variance of Class Embeddings")
    plt.title("Intra-Subset Variance")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved Intra-Subset Variance plot to {output_path}")
    plt.close()

##### 7. Feature Space Coverage Plot
def plot_feature_space_coverage(rand_feature_coverage, knn_d_feature_coverage, knn_s_feature_coverage, output_path="feature_space_coverage.png"):
    # Prepare data for seaborn
    data = {"Method": ["ModelNet-R", "ModelNet-D", "ModelNet-S"], 
            "Coverage": [rand_feature_coverage, knn_d_feature_coverage, knn_s_feature_coverage]}
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Method", y="Coverage", palette=["orange", "blue", "green"])
    plt.ylabel("Trace of Covariance Matrix")
    plt.title("Feature Space Coverage")
    plt.ylim(75, 100)  # Set y-axis limits
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved Feature Space Coverage plot to {output_path}")
    plt.close()



'''
def plot_class_coverage(knn_coverage, rand_coverage, output_path="class_coverage.png"):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["KNN", "Random"], y=[knn_coverage, rand_coverage], palette=["blue", "orange"])
    plt.ylabel("Number of Unique Classes")
    plt.title("Class Coverage Across Subsets")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Class Coverage plot to {output_path}")
    plt.close()
'''
##########------------------Visualization------------------##########




#############------------Evaluation------------#############
train_dir = "/home/ar/FLOCKD/NN_Classification/cifar100/train"
ModelNet_r_dir = "/home/ar/FLOCKD/NN_Classification/train_1-100_5000-15_cifar100_shuffle"
ModelNet_d_dir = "/home/ar/FLOCKD/NN_Classification/train_1-100_5000-15_cifar100_KNN"
ModelNet_s_dir = "/home/ar/FLOCKD/NN_Classification/train_1-100_5000-15_cifar100_similar_mod"
save_path = "/home/ar/FLOCKD/NN_Classification/results/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.fc = torch.nn.Identity()

##### 1. ####--- Compute average pairwise cosine distance 
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class_embeddings = extract_class_embeddings(train_dir, model, transform, device)

rand_diversity = compute_average_subset_diversity(ModelNet_r_dir, class_embeddings)
knn_d_diversity = compute_average_subset_diversity(ModelNet_d_dir, class_embeddings)
knn_s_diversity = compute_average_subset_diversity(ModelNet_s_dir, class_embeddings)
compare_distributions(rand_diversity, 
                      knn_d_diversity, 
                      knn_s_diversity,
                      title="Subset Class Embedding Diversity",
                      xlabel="Avg Pairwise Cosine Distance Between Class Embeddings",
                      output_path=os.path.join(save_path, "subset_diversity.pdf"))


##### 2. ####--- Compute class occurrences
rand_counts = compute_class_occurrences(ModelNet_r_dir)
knn_d_counts = compute_class_occurrences(ModelNet_d_dir)
knn_s_counts = compute_class_occurrences(ModelNet_s_dir)
plot_class_occurrence_histogram(rand_counts, knn_d_counts, knn_s_counts, output_path=os.path.join(save_path, "class_occurrence_histogram.pdf"))
#plot_class_occurrence_histogram1(rand_counts, knn_d_counts, knn_s_counts, output_path=os.path.join(save_path, "class_occurrence_histogram1.png"))

 
##### 3. t-SNE Plot for embeddingds
plot_tsne_triple_subplots(rand_dir=ModelNet_r_dir, 
                          knn_d_dir=ModelNet_d_dir, 
                          knn_s_dir=ModelNet_s_dir, 
                          class_embeddings=class_embeddings, 
                          output_path=os.path.join(save_path, "tsne_plot.pdf"))
'''  
#plot_pca_tsne_for_subsets(rand_dir=rand_dir,
#                          knn_d_dir=knn_d_dir,
#                          knn_s_dir=knn_s_dir,
#                          class_embeddings=class_embeddings,
#                          output_path=os.path.join(save_path, "tsne_plot.png"))


###Previous attempt
plot_tsne_for_subsets(knn_dir=knn_d_dir, 
                      rand_dir=rand_dir, 
                      class_embeddings=class_embeddings, 
                      transform=transform, 
                      device=device, 
                      model=model, 
                      output_path="tsne_plot.png")

plot_tsne_for_subsets_improved(knn_dir=knn_d_dir, 
                               rand_dir=rand_dir, 
                               class_embeddings=class_embeddings, 
                               transform=transform, 
                               device=device, 
                               model=model,
                               output_path=os.path.join(save_path, "tsne_improved.png"))
'''
#run_tsne_for_variants_cached(ModelNet_r_dir, ModelNet_d_dir,  ModelNet_s_dir, output_path=os.path.join(save_path, "tsne_plot1.png"))

##### 4. ####--- Jaccard Similarity
jaccard_scores_D_R = compute_jaccard_similarity(ModelNet_d_dir, ModelNet_r_dir)
jaccard_scores_S_R = compute_jaccard_similarity(ModelNet_s_dir, ModelNet_r_dir)
jaccard_scores_D_S = compute_jaccard_similarity(ModelNet_d_dir, ModelNet_s_dir)
plot_jaccard_similarity_comparison(jaccard_scores_D_R,  
                                   jaccard_scores_S_R, 
                                   jaccard_scores_D_S, 
                                   labels=("ModelNet-D vs ModelNet-R", "ModelNet-S vs ModelNet-R", "ModelNet-D vs ModelNet-S"), 
                                   output_path=os.path.join(save_path, "jaccard_similarity.pdf"))


##### 5. ####--- Subset Redundancy
rand_subset_redundancy = compute_subset_redundancy(ModelNet_r_dir)
knn_d_subset_redundancy = compute_subset_redundancy(ModelNet_d_dir)
knn_s_subset_redundancy = compute_subset_redundancy(ModelNet_s_dir)
plot_subset_redundancy(rand_subset_redundancy, knn_d_subset_redundancy, knn_s_subset_redundancy, os.path.join(save_path, "subset_redundancy.pdf"))


##### 6. ####--- Intra-Subset Variance
rand_intra_variance = compute_intra_subset_variance(ModelNet_r_dir, class_embeddings) #
knn_d_intra_variance = compute_intra_subset_variance(ModelNet_d_dir, class_embeddings) #
knn_s_intra_variance = compute_intra_subset_variance(ModelNet_s_dir, class_embeddings) #
plot_intra_subset_variance(rand_intra_variance, knn_d_intra_variance, knn_s_intra_variance, os.path.join(save_path, "intra_subset_variance.pdf"))


##### 7. ####--- Feature Space Coverage
rand_feature_coverage = compute_feature_space_coverage(ModelNet_r_dir, class_embeddings)
knn_d_feature_coverage = compute_feature_space_coverage(ModelNet_d_dir, class_embeddings)
knn_s_feature_coverage = compute_feature_space_coverage(ModelNet_s_dir, class_embeddings)
plot_feature_space_coverage(rand_feature_coverage, knn_d_feature_coverage, knn_s_feature_coverage, os.path.join(save_path, "feature_space_coverage.pdf"))

'''
####--- Class Coverage
knn_d_class_coverage = compute_class_coverage(knn_d_dir)
rand_class_coverage = compute_class_coverage(rand_dir)
plot_class_coverage(knn_d_class_coverage, rand_class_coverage, os.path.join(save_path, "class_coverage.pdf"))
'''

#############------------Evaluation------------#############
