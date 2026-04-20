from torch.utils.data import WeightedRandomSampler
import pandas as pd
import numpy as np
from collections import Counter

# parameter alpha indicates the strength of the weight shift
# between 0 (sample-uniform) and -1 (typeline-uniform)
def get_sampler(data, alpha=-0.5):
    type_list = data["types"].fillna("")
    typeline_counts = Counter(type_list)
    typelines = list(typeline_counts.keys())

    # build weights by typeline
    typeline_freq = np.array([typeline_counts[t] for t in typelines])
    typeline_freq = typeline_freq / typeline_freq.sum()
    typeline_weights = typeline_freq ** alpha
    typeline_weights = typeline_weights / typeline_weights.sum()
    weights_dict = dict(zip(typelines, typeline_weights))

    weights = []
    for sample in type_list:
        weights.append(weights_dict[sample])
    
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)