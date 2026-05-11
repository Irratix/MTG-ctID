from collections import Counter
import pandas as pd
import re

MIN_COUNT = 100

def build_vocab(df):
    vocab = {
        "card_type": card_type_vocab(df),
        "creature_type": creature_type_vocab(df),
        "mana_cost": mana_cost_vocab(df),
        "colors": colors_vocab(df),
        "keywords": keywords_vocab(df),
        "rarity": rarity_vocab(df),
        "p/t": pt_vocab(df)
    }
    # print(vocab)
    return vocab

def card_type_vocab(df):
    # iterative histogram filter
    df = df.copy()
    hist = Counter(
        t for type_str in df["card_type"] for t in type_str.split("|")
    )
    while min(hist.values()) < MIN_COUNT:
        valid_types = {t for t, count in hist.items() if count >= MIN_COUNT}
        df = df[df["card_type"].apply(lambda s: all(t in valid_types for t in s.split("|")))]    
        hist = Counter(
            t for type_str in df["card_type"] for t in type_str.split("|")
        )

    valid_types = sorted(t for t, _ in hist.items())

    return {t: i for i, t in enumerate(valid_types)}

def creature_type_vocab(df):
    # iterative histogram filter
    df = df.copy()
    hist = Counter(
        t for type_str in df["creature_type"] if not pd.isna(type_str) for t in type_str.split("|")
    )
    while min(hist.values()) < MIN_COUNT:
        valid_types = {t for t, count in hist.items() if count >= MIN_COUNT}
        df = df[df["creature_type"].apply(lambda s: pd.isna(s) or all(t in valid_types for t in s.split("|")))]    
        hist = Counter(
            t for type_str in df["creature_type"] if not pd.isna(type_str) for t in type_str.split("|")
        )

    valid_types = sorted(t for t, _ in hist.items())

    return {t: i for i, t in enumerate(valid_types)}

def mana_cost_vocab(df):
    df = df.copy()
    hist = Counter(
        symbol for cost in df["mana_cost"].dropna() for symbol in re.findall(r'\{([^}]+)\}', cost)
        )
    while min(hist.values()) < MIN_COUNT:
        valid_symbols = {s for s, count in hist.items() if count >= MIN_COUNT}
        df = df[df["mana_cost"].apply(
            lambda cost: not pd.isna(cost) and all(s in valid_symbols for s in re.findall(r'\{([^}]+)\}', cost))
            )]    
        hist = Counter(
            c for cost in df["mana_cost"] if not pd.isna(cost) for c in re.findall(r'\{([^}]+)\}', cost)
        )

    valid_symbols = {s for s, count in hist.items() if count >= MIN_COUNT}
    return {c: i for i, c in enumerate(valid_symbols)}

def colors_vocab(df):
    colors = sorted(set(
        c for c_str in df["colors"] if not pd.isna(c_str) for c in c_str.split('|')
    ))
    return {t: i for i, t in enumerate(colors)}

def keywords_vocab(df):
    # iterative histogram filter
    df = df.copy()
    hist = Counter(
        t for k_str in df["keywords"] if not pd.isna(k_str) for t in k_str.split("|")
    )
    while min(hist.values()) < MIN_COUNT:
        valid_keywords = {t for t, count in hist.items() if count >= MIN_COUNT}
        df = df[df["keywords"].apply(lambda s: pd.isna(s) or all(t in valid_keywords for t in s.split("|")))]    
        hist = Counter(
            t for k_str in df["keywords"] if not pd.isna(k_str) for t in k_str.split("|")
        )

    valid_keywords = sorted(t for t, _ in hist.items())

    return {t: i for i, t in enumerate(valid_keywords)}

def rarity_vocab(df):
    hist = Counter(
        r for r in df["rarity"]
    )
    valid_rarity = sorted(t for t, count in hist.items() if count >= MIN_COUNT)
    
    return {r: i for i, r in enumerate(valid_rarity)}

def pt_vocab(df):
    return {
        "power": 0,
        "toughness": 1
    }