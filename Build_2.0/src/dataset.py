from torch.utils.data import Dataset

from PIL import Image
import pandas as pd
from pathlib import Path
import torch
import re
from tqdm import tqdm

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
        for i in tqdm(range(len(manifest))):
            record = manifest.iloc[i]
            labels.append({
                "creature_types": self.encode_creature_types(record),
                "colors": self.encode_colors(record),
                "card_types": self.encode_card_types(record),
                "keywords": self.encode_keywords(record),
                "rarity": self.encode_rarity(record),
                "mana_cost": self.encode_mana_cost(record),
                "pt": self.encode_pt(record)
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
    
    def encode_mana_cost(self, record):
        label = torch.zeros(len(self.vocab["mana_cost"]))
        if not pd.isna(record["mana_cost"]):
            for c in re.findall(r'\{([^}]+)\}', record["mana_cost"]):
                if c in self.vocab["mana_cost"]:
                    label[self.vocab["mana_cost"][c]] += 1.0
        return label
    
    def encode_pt(self, record):
        label = torch.zeros(2)
        if record["power"].isnumeric():
            label[0] = float(record["power"])
        if record["toughness"].isnumeric():
            label[1] = float(record["toughness"])
        return label