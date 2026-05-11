from torch.utils.data import Dataset

from PIL import Image
import pandas as pd
from pathlib import Path
import torch

class MTGDataset(Dataset):
    def __init__(self, manifest, vocab):
        self.vocab = vocab
        self.paths = manifest["path"].tolist()
        self.labels = self.compute_labels(manifest)

    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        image = Image.open(Path(self.paths[idx])).convert("RGB")
        labels = self.labels[idx]
        return image, labels
    
    def compute_labels(self, manifest):
        labels = []
        for i in range(len(manifest)):
            record = manifest.iloc[i]
            labels.append({
                "creature_types": self.encode_creature_types(record),
                "colors": self.encode_colors(record),
                "card_types": self.encode_card_types(record),
                "keywords": self.encode_keywords(record),
                "rarity": self.encode_rarity(record),
                "mana_cost": self.encode_mana_cost(record),
                "pt": self.encode_pt(record),
                "creature_type_mask": self.get_creature_type_mask(record),
                # TODO: determine masks for other classes
            })
        return labels

    ##### vector encoders #####
    def encode_creature_types(self, record):
        label = torch.zeros(len(self.vocab["creature_type"]))
        types = record["creature_type"] if not pd.isna(record["creature_type"]) else []
        for t in types:
            if t in self.vocab["creature_type"]:
                label[self.vocab["creature_type"][t]] = 1.0
        return label
    
    def encode_colors(self, record):
        label = torch.zeros(len(self.vocab["colors"]))
        types = record["colors"] if not pd.isna(record["colors"]) else []
        for t in types:
            if t in self.vocab["colors"]:
                label[self.vocab["colors"][t]] = 1.0
        return label
    
    def encode_card_types(self, record):
        label = torch.zeros(len(self.vocab["card_type"]))
        types = record["card_type"] if not pd.isna(record["card_type"]) else []
        for t in types:
            if t in self.vocab["card_type"]:
                label[self.vocab["card_type"][t]] = 1.0
        return label
    
    def encode_keywords(self, record):
        label = torch.zeros(len(self.vocab["keywords"]))
        types = record["keywords"] if not pd.isna(record["keywords"]) else []
        for t in types:
            if t in self.vocab["keywords"]:
                label[self.vocab["keywords"][t]] = 1.0
        return label
    
    def encode_rarity(self, record):
        label = torch.zeros(len(self.vocab["rarity"]))
        if record["rarity"] in self.vocab["rarity"]:
            label[self.vocab["rarity"][record["rarity"]]] = 1.0
        return label
    
    # TODO
    def encode_mana_cost(self, record):
        ...
    
    # TODO
    def encode_pt(self, record):
        ...

    ##### masks #####
    # TODO
    def get_creature_type_mask(self, record):
        ...