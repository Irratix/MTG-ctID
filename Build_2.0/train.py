import pandas as pd
from src.vocab import build_vocab
from src.dataset import MTGDataset
from src.model import MTGModel

CSV_PATH = "Build_2.0/data/all_cards_manifest.csv"

def main():
    print("===== loading manifest =====")
    df = pd.read_csv(CSV_PATH)
    print(f"Total cards: {len(df)}")

    print("===== building vocab =====")
    vocab = build_vocab(df)

    print("===== building dataset =====")
    dataset = MTGDataset(df, vocab)

    print("===== building base model =====")
    model = MTGModel(vocab)

if __name__ == "__main__":
    main()
