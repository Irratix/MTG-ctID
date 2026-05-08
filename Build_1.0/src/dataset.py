from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import  v2
import matplotlib.pyplot as plt
import numpy as np

# data augmentation
train_transform = v2.Compose([
                # shape
                v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(),

                # color
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                v2.RandomGrayscale(p=0.1),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

val_transform = v2.Compose([
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

# dataset class
class CreatureDataset(Dataset):
    def __init__(self, df, all_types, is_training=True):
        self.df = df.reset_index(drop=True)
        self.all_types = all_types
        self.type_to_idx = {t: i for i, t in enumerate(all_types)}
        self.transform = train_transform if is_training else val_transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image = Image.open(row["image_path"].replace("\\", "/")).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        label = torch.zeros(len(self.all_types))
        types = row["types"].split("|") if isinstance(row["types"], str) else []
        for t in types:
            if t in self.all_types:
                label[self.type_to_idx[t]] = 1.0

        return image, label


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return tensor * std + mean

def show_transform(x=3, y=3, training=True):
    # Load one image
    img = Image.open("data/all_cards/3cf292f8-161b-48bd-aa74-f8e3783e80c2.jpg").convert("RGB")

    # Generate augmented samples
    num_samples = x * y
    transform = train_transform if training else val_transform
    augmented = [train_transform(img) for _ in range(num_samples)]

    # Plot
    fig, axes = plt.subplots(y, x, figsize=(x * 2, y * 2))

    for i, ax in enumerate(axes.flat):
        img_tensor = denormalize(augmented[i]).clamp(0, 1)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
