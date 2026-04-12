import requests
import json
import argparse
from pathlib import Path
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


SET = "oracle_cards"
FILTER_SETTINGS = {
    "min_of_ctype": 100,
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


# get a record of all cards (pre-histogram filter) that could be in the dataset
def get_records():
    with open(f"data/{SET}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = []
    seen = set()
    for card in data:
        # Some of these are just to stop scanning for double-sided cards, we may want to support those later
        if "illustration_id" not in card:
            continue
        if "//" in card["name"]:
            continue
        if card["illustration_id"] in seen:
            continue
        ctype = get_creature_type(card["type_line"])
        if not ctype["is_creature"]:
            continue
        if card["layout"] not in FILTER_SETTINGS["allowed_layout"]:
            continue
        if card.get("image_status") not in ("highres_scan", "lowres"):
            continue
        if card["legalities"]["vintage"] == "not_legal":
            continue
        if "type_line" not in card:
            continue

        illustration_id = card["illustration_id"]
        records.append((illustration_id, ctype["types"], card["name"], Path(f"data/{SET}/{illustration_id}.jpg"), card["image_uris"]["art_crop"]))
        seen.add(illustration_id)
    
    return records
        

# filter records to support minimum frequency of creature types
def hist_filter(records):
    # we keep looping this until we have a return value
    hist = Counter()
    for _, ctype, _, _, _ in records:
        hist.update(ctype)
    
    # if the rarest creature type 
    if min(hist.values()) >= FILTER_SETTINGS["min_of_ctype"]:
        return records

    new_records = []

    for illustration_id, ctype, name, path, uri in records:
        if any(hist[t] < FILTER_SETTINGS["min_of_ctype"] for t in ctype):
            continue
        new_records.append((illustration_id, ctype, name, path, uri))
    
    print(f"after iteration of histogram filter, {len(new_records)} illustrations remain")
    return hist_filter(new_records)


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
def get_labeled_data(records):
    path_csv = Path(f"data/{SET}_manifest.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["illustration_id", "types", "name", "image_path"])
        for illustration_id, ctype, name, path, _ in records:
            writer.writerow([illustration_id, "|".join(ctype), name, path])

    # collect cards and write data
    download_tasks = [
        (illustration_id, uri)
        for illustration_id, _, _, _, uri in records
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
    print(f"-- getting {SET} dictionary --")
    get_card_dict(refresh=args.refresh)
    print(f"-- building creature card records --")
    records = get_records()
    print(*records[0:10], sep="\n")
    print(len(records))
    print(f"-- performing histogram filter --")
    records = hist_filter(records)
    print(f"-- downloading images --")
    get_labeled_data(records)


if __name__ == "__main__":
    main()