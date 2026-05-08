import requests
import ijson
import argparse
from pathlib import Path
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

SET = "all_cards"
FILTER_SETTINGS = {
    "min_of_ctype": 100,
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


### what follows are helper functions to parse different layouts properly ###
# supports "normal", "mutate", "leveler", "meld", "prototype"
def get_record_normal(card):
    if "illustration_id" not in card:
        return []
    illustration_id = card["illustration_id"]
    ctype = get_creature_type(card["type_line"])
    if not ctype["is_creature"]:
        return []
    if card.get("image_status") not in ("highres_scan", "lowres"):
        return []
    if card["legalities"]["vintage"] == "not_legal":
        return []

    return [(illustration_id, ctype["types"], card["name"], Path(f"data/{SET}/{illustration_id}.jpg"), card["image_uris"]["art_crop"])]


# supports "token"
def get_record_token(card):
    if "illustration_id" not in card:
        return []
    illustration_id = card["illustration_id"]
    ctype = get_creature_type(card["type_line"])
    if not ctype["is_creature"]:
        return []
    if card.get("image_status") not in ("highres_scan", "lowres"):
        return []

    return [(illustration_id, ctype["types"], card["name"], Path(f"data/{SET}/{illustration_id}.jpg"), card["image_uris"]["art_crop"])]


# supports "transform", "modal_dfc"
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
        ctype = get_creature_type(face["type_line"])
        if not ctype["is_creature"]:
            continue
        records.append((illustration_id, ctype["types"], face["name"], Path(f"data/{SET}/{illustration_id}.jpg"), face["image_uris"]["art_crop"]))
    
    return records


# supports "adventure", "prepare"
def get_record_adventure(card):
    if "illustration_id" not in card:
        return []
    illustration_id = card["illustration_id"]
    ctype = get_creature_type(card["type_line"].split(" // ")[0])
    if not ctype["is_creature"]:
        return []
    if card.get("image_status") not in ("highres_scan", "lowres"):
        return []

    return [(illustration_id, ctype["types"], card["name"].split(" // ")[0], Path(f"data/{SET}/{illustration_id}.jpg"), card["image_uris"]["art_crop"])]


# supports "double_faced_token"
def get_record_double_faced_token(card):
    if card["image_status"] not in ("highres_scan", "lowres"):
        return []
    
    records = []
    for face in card["card_faces"]:
        if "illustration_id" not in face:
            continue
        illustration_id = face["illustration_id"]
        ctype = get_creature_type(face["type_line"])
        if not ctype["is_creature"]:
            continue
        records.append((illustration_id, ctype["types"], face["name"], Path(f"data/{SET}/{illustration_id}.jpg"), face["image_uris"]["art_crop"]))
    
    return records


# get a record of all cards (pre-histogram filter) that could be in the dataset
def get_records():
    records = []
    seen = set()

    with open(f"data/{SET}.json", "rb") as f:
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
            elif card["layout"] in ("double_faced_token"):
                record = get_record_double_faced_token(card)
            else:
                continue

            # append to list of records where applicable
            for illustration_id, ctype, name, path, uri in record:
                if illustration_id in seen:
                    continue
                records.append((illustration_id, ctype, name, path, uri))
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
    
    print(f"after iteration of histogram filter, {len(new_records)} illustrations with {len(hist)} unique creature types remain")
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
    print(*records[0:10], sep="\n")
    print(len(records))
    print(f"-- performing histogram filter --")
    records = hist_filter(records)
    print(f"-- downloading images --")
    get_labeled_data(records)


if __name__ == "__main__":
    main()