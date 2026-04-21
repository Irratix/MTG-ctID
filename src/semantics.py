import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

path = "data/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"

def plot_similarity_matrix(matrix, types):
    # convert dict to 2D array in types order
    arr = np.array([matrix[t] for t in types])
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(arr, cmap='viridis')
    
    ax.set_xticks(range(len(types)))
    ax.set_yticks(range(len(types)))
    ax.set_xticklabels(types, rotation=90, fontsize=7)
    ax.set_yticklabels(types, fontsize=7)
    
    plt.colorbar(im, ax=ax)
    plt.title("Semantic Similarity Matrix")
    plt.tight_layout()
    plt.show()

def read_glove():
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading GloVe data"):
            values = line.split(" ")
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_smoothing_matrix(types):
    embeddings = read_glove()
    matrix = {}
    for idx, word1 in enumerate(types):
        if word1.lower() not in embeddings:
            vec = np.zeros(len(types))
            vec[idx] = 1.0
            matrix[word1] = vec
            continue
        vec = []
        for word2 in types:
            if word2.lower() not in embeddings:
                vec.append(0.0)
            else:
                vec.append(cosine_similarity(
                    embeddings[word1.lower()], 
                    embeddings[word2.lower()]
                ))
        vec = np.array(vec)
        vec = ((vec + 1) / 2) ** 2 / 5
        vec[idx] = 1.0
        matrix[word1] = vec
    return matrix