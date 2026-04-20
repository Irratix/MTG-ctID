from torch.utils.data import WeightedRandomSampler
import numpy as np

def compute_sample_weights(df, all_types, type_counts, normalized, cap_ratio=3.0):
    weights = []
    for _, row in df.iterrows():
        types = row["types"].split("|") if isinstance(row["types"], str) else []
        
        # base weight from inverse frequency
        base = sum(1.0 / type_counts[t] for t in types if t in type_counts)
        if not base:
            base = 1.0
        
        # correlation penalty from pairwise co-occurrence
        if len(types) > 1:
            pairs = [
                normalized[all_types.index(t1)][all_types.index(t2)]
                for t1 in types for t2 in types
                if t1 != t2 and t1 in all_types and t2 in all_types
            ]
            penalty = 1.0 + sum(pairs) / len(pairs)
        else:
            penalty = 1.0
        
        weights.append(base / penalty)
    
    # apply cap ratio
    min_w = min(weights)
    max_w = min_w * cap_ratio
    weights = [min(w, max_w) for w in weights]
    
    return weights

def get_normalized(data, types):
    n_types = len(types)
    type_to_idx = {t: i for i, t in enumerate(types)}

    cooccurrence = np.zeros((n_types, n_types))
    for _, row in data.iterrows():
        if not isinstance(row["types"], str):
            continue
        types = row["types"].split("|")
        for t1 in types:
            for t2 in types:
                if t1 in type_to_idx and t2 in type_to_idx:
                    cooccurrence[type_to_idx[t1]][type_to_idx[t2]] += 1

    normalized = np.zeros((n_types, n_types))
    for i in range(n_types):
        if cooccurrence[i, i] > 0:
            normalized[i, :] = cooccurrence[i, :] / cooccurrence[i, i]
    
    return normalized

def get_sampler(data, types, type_counts):
    normalized = get_normalized(data, types)
    weights = compute_sample_weights(data, types, type_counts, normalized)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)