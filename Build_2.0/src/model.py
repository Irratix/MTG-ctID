from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torch.nn as nn

class MTGModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

        self.features = backbone.features
        self.pool = backbone.avgpool

        self.heads = nn.ModuleDict({
            'card_type': card_type_head(vocab),
            'creature_type': creature_type_head(vocab),
            'mana_cost': mana_cost_head(vocab),
            'colors': colors_head(vocab),
            'keywords': keywords_head(vocab),
            'rarity': rarity_head(vocab),
            'power/toughness': pt_head(vocab)
        })

        # on initialization, the backbone should be frozen
        for param in self.features.parameters():
            param.requires_grad = False

        # we keep track of which depth has been unfrozen
        self.unfrozen = 0
    
    def forward(self, x, task):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.heads[task](x)

    def unfreeze_next_layer(self):
        self.unfrozen += 1
        for param in self.features[-self.unfrozen:].parameters():
            param.requires_grad = True
    
def card_type_head(vocab):
    return nn.Sequential(
        nn.Linear(1408, 512),
        nn.SiLU(),
        nn.Linear(512, len(vocab['card_type']))
    )

def creature_type_head(vocab):
    return nn.Sequential(
        nn.Linear(1408, 512),
        nn.SiLU(),
        nn.Linear(512, len(vocab['creature_type']))
    )

def mana_cost_head(vocab):
    return nn.Sequential(
        nn.Linear(1408, 512),
        nn.SiLU(),
        nn.Linear(512, len(vocab['mana_cost']))
    )

def colors_head(vocab):
    return nn.Sequential(
        nn.Linear(1408, 256),
        nn.SiLU(),
        nn.Linear(256, len(vocab['colors']))
    )

def keywords_head(vocab):
    return nn.Sequential(
        nn.Linear(1408, 512),
        nn.SiLU(),
        nn.Linear(512, len(vocab['keywords']))
    )

def rarity_head(vocab):
    return nn.Sequential(
        nn.Linear(1408, 256),
        nn.SiLU(),
        nn.Linear(256, len(vocab['rarity']))
    )

def pt_head(vocab):
    return nn.Sequential(
        nn.Linear(1408, 512),
        nn.SiLU(),
        nn.Linear(512, len(vocab['p/t']))
    )