from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from src.models.modelconfig import ModelConfig
import torch
import torch.nn as nn

def build_efficientnet_b2(types, device):
    # Model choice
    en_b2 = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    num_features = en_b2.classifier[1].in_features
    en_b2.classifier[1] = nn.Linear(num_features, len(types))
    en_b2 = en_b2.to(device)    

    # we freeze all except the last layer
    for param in en_b2.parameters():
        param.requires_grad = False
    for param in en_b2.classifier.parameters():
        param.requires_grad = True  

    # Optimize
    optimizer = torch.optim.Adam([
        {'params': en_b2.features[6].parameters(), 'lr': 1e-5},
        {'params': en_b2.features[7].parameters(), 'lr': 2e-5},
        {'params': en_b2.features[8].parameters(), 'lr': 5e-5},
        {'params': en_b2.classifier.parameters(), 'lr': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10) 

    return ModelConfig(
        name="EfficientNet_b2",
        model=en_b2, 
        optimizer=optimizer, 
        scheduler=scheduler,
        checkpoint_folder="trained_models/EfficientNet_b2",
        num_epochs=50,
        unfreeze_schedule={10:en_b2.features[8], 20:en_b2.features[7], 30:en_b2.features[6]}
    )