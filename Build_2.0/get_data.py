import requests
import ijson
import argparse
from pathlib import Path
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

SET = "all_cards"
FOLDER = f"Build_2.0/data"


# gets the bulk data dictionary of all cards
def get_card_dict(refresh=False):
    bulk_path = Path(f"{FOLDER}/{SET}.json")
    bulk_path.parent.mkdir(parents=True, exist_ok=True)

    if bulk_path.exists() and not refresh:
        print("Card data already exists, skipping download. Use --refresh to re-download anyways.")
        print("You may want to do this if your card data is either corrupted or outdated.")
        return

    response = requests.get("https://api.scryfall.com/bulk-data")
    response.raise_for_status()

    uri = ""
    for obj in response.json()["data"]:
        if obj["type"] == SET:
            uri = obj["download_uri"]

    response = requests.get(uri, stream=True)
    response.raise_for_status()
    with open(bulk_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


# parse the creature type if the card is a creature
def get_creature_type(typeline):
    typeline = typeline.replace("Time Lord", "Time-Lord")
    types = typeline.split(" — ")
    if len(types) == 1:
        return []
    return types[1].split(" ")


# parse the card type
def get_card_type(typeline):
    types = typeline.split(" — ")[0]
    return types.split(" ")
    

### what follows are helper functions to parse different layouts properly ###
# supports "normal", "mutate", "leveler", "meld", "prototype"
def get_record_normal(card):
    if "illustration_id" not in card:
        return []
    illustration_id = card["illustration_id"]
    card_type = get_card_type(card["type_line"])
    if "Creature" not in card_type:
        return []
    if card.get("image_status") not in ("highres_scan", "lowres"):
        return []
    if card["legalities"]["vintage"] == "not_legal":
        return []

    creature_type = get_creature_type(card["type_line"])

    return [{
        "name": card["name"], 
        "illustration_id": illustration_id, 
        "path": Path(f"{FOLDER}/{SET}/{illustration_id}.jpg"), 
        "uri": card["image_uris"]["art_crop"],
        "card_type": card_type,
        "creature_type": creature_type,
        "mana_cost": card["mana_cost"],
        "colors": card["colors"],
        "keywords": card["keywords"],
        "rarity": card["rarity"],
        "power": card["power"],
        "toughness": card["toughness"]
        }]


# supports "token"
def get_record_token(card):
    if "illustration_id" not in card:
        return []
    illustration_id = card["illustration_id"]
    card_type = get_card_type(card["type_line"])
    if "Creature" not in card_type:
        return []
    if card.get("image_status") not in ("highres_scan", "lowres"):
        return []

    creature_type = get_creature_type(card["type_line"])

    return [{
        "name": card["name"], 
        "illustration_id": illustration_id, 
        "path": Path(f"{FOLDER}/{SET}/{illustration_id}.jpg"), 
        "uri": card["image_uris"]["art_crop"],
        "card_type": card_type,
        "creature_type": creature_type,
        "mana_cost": card["mana_cost"],
        "colors": card["colors"],
        "keywords": card["keywords"],
        "rarity": card["rarity"],
        "power": card["power"],
        "toughness": card["toughness"]
        }]


# supports "transform", "modal_dfc", "reversible_card"
def get_record_transform(card):
    if card["image_status"] not in ("highres_scan", "lowres"):
        return []
    if card["legalities"]["vintage"] == "not_legal":
        return []
    
    records = []
    for face in card["card_faces"]:
        if "illustration_id" not in face:
            continue
        illustration_id = face["illustration_id"]
        card_type = get_card_type(face["type_line"])
        if "Creature" not in card_type:
            continue

        creature_type = get_creature_type(face["type_line"])
        keywords = [keyword for keyword in card["keywords"] if keyword in face["oracle_text"]]

        records.append({
            "name": face["name"], 
            "illustration_id": illustration_id, 
            "path": Path(f"{FOLDER}/{SET}/{illustration_id}.jpg"), 
            "uri": face["image_uris"]["art_crop"],
            "card_type": card_type,
            "creature_type": creature_type,
            "mana_cost": face["mana_cost"],
            "colors": face["colors"],
            "keywords": keywords,
            "rarity": card["rarity"],
            "power": face["power"],
            "toughness": face["toughness"]
            })
    
    return records


# supports "adventure", "prepare"
def get_record_adventure(card):
    if "illustration_id" not in card:
        return []
    illustration_id = card["illustration_id"]
    card_type = get_card_type(card["type_line"].split(" // ")[0])
    if "Creature" not in card_type:
        return []
    if card.get("image_status") not in ("highres_scan", "lowres"):
        return []
    if card["legalities"]["vintage"] == "not_legal":
        return []

    creature_type = get_creature_type(card["type_line"].split(" // ")[0])

    return [{
        "name": card["name"], 
        "illustration_id": illustration_id, 
        "path": Path(f"{FOLDER}/{SET}/{illustration_id}.jpg"), 
        "uri": card["image_uris"]["art_crop"],
        "card_type": card_type,
        "creature_type": creature_type,
        "mana_cost": card["mana_cost"].split(" // ")[0],
        "colors": card["colors"],
        "keywords": card["keywords"],
        "rarity": card["rarity"],
        "power": card["power"],
        "toughness": card["toughness"]
        }]


# supports "double_faced_token"
def get_record_double_faced_token(card):
    if card["image_status"] not in ("highres_scan", "lowres"):
        return []
    
    records = []
    for face in card["card_faces"]:
        if "illustration_id" not in face:
            continue
        illustration_id = face["illustration_id"]
        card_type = get_card_type(face["type_line"])
        if "Creature" not in card_type:
            continue

        creature_type = get_creature_type(face["type_line"])
        keywords = [keyword for keyword in card["keywords"] if keyword in face["oracle_text"]]

        records.append({
            "name": face["name"], 
            "illustration_id": illustration_id, 
            "path": Path(f"{FOLDER}/{SET}/{illustration_id}.jpg"), 
            "uri": face["image_uris"]["art_crop"],
            "card_type": card_type,
            "creature_type": creature_type,
            "mana_cost": face["mana_cost"],
            "colors": face["colors"],
            "keywords": keywords,
            "rarity": card["rarity"],
            "power": face["power"],
            "toughness": face["toughness"]
            })
    
    return records


# get a record of all cards (pre-histogram filter) that could be in the dataset
def get_records():
    records = []
    seen = set()

    with open(f"{FOLDER}/{SET}.json", "rb") as f:
        for card in ijson.items(f, "item"):
            # get record for individual card
            if card["layout"] in ("normal", "mutate", "leveler", "meld", "prototype"):
                record = get_record_normal(card)
            elif card["layout"] in ("token"):
                record = get_record_token(card)
            elif card["layout"] in ("transform", "modal_dfc", "reversible_card"):
                record = get_record_transform(card)
            elif card["layout"] in ("adventure", "prepare"):
                record = get_record_adventure(card)
            elif card["layout"] == "double_faced_token":
                record = get_record_double_faced_token(card)
            else:
                continue

            # append to list of records where applicable
            for face in record:
                if face["illustration_id"] in seen:
                    continue
                records.append(face)
                seen.add(face["illustration_id"])
    
    return records


# builds csv with all data labels
def get_data_labels(records):
    path_csv = Path(f"{FOLDER}/{SET}_manifest.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["name", "illustration_id", "path", "uri", "card_type", "creature_type", "mana_cost", "colors", "keywords", "rarity", "power", "toughness"])
        for face in records:
            writer.writerow([
                face["name"], 
                face["illustration_id"],
                face["path"],
                face["uri"],
                "|".join(face["card_type"]),
                "|".join(face["creature_type"]), 
                face["mana_cost"],
                "|".join(face["colors"]),
                "|".join(face["keywords"]),
                face["rarity"],
                face["power"],
                face["toughness"]
                ])


# downloads an image
def download_image(args):
    image_uri, image_path = args
    path_img = Path(image_path)
    if path_img.exists():
        return
    path_img.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(image_uri, stream=True)
    response.raise_for_status()
    with open(path_img, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def get_images(records):
    # collect cards and write data
    download_tasks = [
        (face["uri"], face["path"])
        for face in records
    ]

    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            list(tqdm(executor.map(download_image, download_tasks), total=len(download_tasks)))
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down...")
        executor.shutdown(wait=False, cancel_futures=True)

# parse setting arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Download MtG card data and artwork")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download bulk card data even if it already exists."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"-- getting {SET} dictionary --")
    get_card_dict(refresh=args.refresh)
    print(f"-- building creature card records --")
    records = get_records()
    # print(*records[0:10], sep="\n")
    print(len(records))
    print(f"-- building data manifest --")
    get_data_labels(records)
    print(f"-- downloading images --")
    get_images(records)


if __name__ == "__main__":
    main()