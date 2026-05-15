import pandas as pd
from src.vocab import build_vocab
from src.dataset import MTGDataset
from src.model import MTGModel
from src.loss import loss
import random
from torch.utils.data import DataLoader

CSV_PATH = "Build_2.0/data/all_cards_manifest.csv"

def main():
    print("===== loading manifest =====")
    df = pd.read_csv(CSV_PATH)
    print(f"Total cards: {len(df)}")

    print("===== building vocab =====")
    vocab = build_vocab(df)

    print("===== building dataset =====")
    indices = list(range(len(df)))
    split = int(0.8 * len(df))
    random.shuffle(indices)
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_loader = DataLoader(
        MTGDataset(df.iloc[train_indices], vocab),
        batch_size=64,
        num_workers=6,
        pin_memory_device="cuda",
        persistent_workers=True
    )
    val_loader = DataLoader(
        MTGDataset(df.iloc[val_indices], vocab),
        batch_size=64,
        num_workers=6,
        pin_memory_device="cuda",
        persistent_workers=True
    )


    print("===== building base model =====")
    model = MTGModel(vocab)

    loss_fn = loss

if __name__ == "__main__":
    main()
