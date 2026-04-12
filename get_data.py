import requests
import json
import argparse
from pathlib import Path
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# TODO: Filter out "not_legal" in Vintage to get rid of test cards and un-cards properly

SET = "all_cards"
FILTER_SETTINGS = {
    "min_of_ctype": 100,
    "banned_sets": {"uno", "unh", "ung", "unf"},
    "banned_types": {"Saga"},
    "allowed_layout": {"normal"}
}

# gets the bulk data dictionary of all cards
def get_card_dict(refresh=False):
    bulk_path = Path(f"data/{SET}.json")
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


# parse the card and creature type
def get_creature_type(typeline):
    typeline = typeline.replace("Time Lord", "Time-Lord")
    types = typeline.split(" — ")
    if "Creature" not in types[0]:
        return {
            "is_creature": False,
            "types": []
        }
    if len(types) == 1:
        return {
            "is_creature": True,
            "types": []
        }
    return {
        "is_creature": True,
        "types": types[1].split(" ")
    }


# checks all properties that determine whether or not a card should be downloaded in the first place
def is_valid_card(card, seen, ctype, hist=Counter()):
    # TODO: handle double-faced cards. The following two checks just makes sure we skip them for now
    if "illustration_id" not in card:
        return False
    if "//" in card["name"]:
        return False
    if card["illustration_id"] in seen:
        return False
    if not ctype["is_creature"]:
        return False
    if card["set"] in FILTER_SETTINGS["banned_sets"]:
        return False
    if card["layout"] not in FILTER_SETTINGS["allowed_layout"]:
        return False
    if any(t in hist and hist[t] < FILTER_SETTINGS["min_of_ctype"] for t in ctype["types"]):
        return False
    if any(t in FILTER_SETTINGS["banned_types"] for t in ctype["types"]):
        return False
    if card.get("image_status") not in ("highres_scan", "lowres"):
        return False
    return True


# downloads an image
def download_image(args):
    illustration_id, image_uri = args
    path_img = Path(f"data/{SET}/{illustration_id}.jpg")
    if path_img.exists():
        return
    path_img.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(image_uri, stream=True)
    response.raise_for_status()
    with open(path_img, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


# gets labeled data based on the bulk card dictionary
def get_labeled_data():
    # open the data
    with open(f"data/{SET}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # collect histogram for the filtering process
    hist = Counter()
    seen = set()
    for card in data:
        print(card["name"])
        # TODO: Figure out: Cards don't have a type_line for some reason?
        # - Jinnie Fay, Jetmir's Second // Jinnie Fay, Jetmir's Second 
        if "type_line" not in card:
            continue
        ctype = get_creature_type(card["type_line"])
        if not is_valid_card(card, seen, ctype):
            continue
        hist.update(ctype["types"])
        seen.add(card["illustration_id"])

    # get a proper record of cards I wish to download
    records = []
    seen = set()
    for card in data:
        # TODO: see above
        if "type_line" not in card:
            continue
        ctype = get_creature_type(card["type_line"])
        if not is_valid_card(card, seen, ctype, hist):
            continue
        seen.add(card["illustration_id"])
        records.append((card, ctype))

    path_csv = Path(f"data/{SET}_manifest.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["illustration_id", "name", "types", "image_path"])
        for card, ctype in records:
            illustration_id = card["illustration_id"]
            path_img = Path(f"data/{SET}/{illustration_id}.jpg")
            writer.writerow([illustration_id, card["name"], "|".join(ctype["types"]), path_img])

    # collect cards and write data
    download_tasks = [
        (card["illustration_id"], card["image_uris"]["art_crop"])
        for card, _ in records
    ]

    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(download_image, download_tasks)
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
    get_card_dict(refresh=args.refresh)
    get_labeled_data()


if __name__ == "__main__":
    main()