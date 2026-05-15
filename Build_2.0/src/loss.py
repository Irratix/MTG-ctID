import torch.nn as nn

loss = {
    'card_type': nn.BCEWithLogitsLoss(),
    'creature_type': nn.BCEWithLogitsLoss(),
    'mana_cost': nn.PoissonNLLLoss(log_input=True),
    'colors': nn.BCEWithLogitsLoss(),
    'keywords': nn.BCEWithLogitsLoss(),
    'rarity': nn.CrossEntropyLoss(),
    'power/toughness': nn.PoissonNLLLoss(log_input=True)
}