import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

new_json_dir = Path("/home/mark/Data/NFT_Dataset/json")
media_dir = Path("/home/mark/Data/NFT_Dataset/media")


def cleanup_1():
    """
    1. Keep filename only for media file paths.
    2. Add id to each file.
    """
    json_dir = Path("/home/mark/CODE/NOT_MINE/project_numpie/data/preprocessed/json")
    new_json_dir = Path("/home/mark/Data/MARK_NFT/json")

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_path = json_dir / filename

            with json_path.open() as f:
                data = json.load(f)
                f.close()

            media_filenames = data.get("media_filenames", [])
            media_filenames = [str(Path(i).name) for i in media_filenames]
            data["id"] = int(Path(filename).stem)
            data["media_filenames"] = media_filenames

            outpath = new_json_dir / f"{data['id']:05}.json"

            with outpath.open("w") as f:
                json.dump(data, f, indent=4)
                f.close()


def cleanup_2():
    """Rename all media files with zero padding."""
    for filename in os.listdir(media_dir):
        filepath = media_dir / filename
        stem = Path(filename).stem
        ext = Path(filename).suffix
        new_name = f"{int(stem):05}{ext}"
        new_path = media_dir / new_name
        filepath.rename(new_path)
        print(f"OK. '{filename}' -> '{new_name}'")


def cleanup_3():
    """
    1. Sort keys of each json file.
    2. rename media filenames with zero padding.
    """

    sorted_keys = [
        "id",
        "name",
        "description",
        "collection_name",
        "collection_description",
        "transaction_time",
        "eth_price",
        "eth_price_decimal",
        "usd_price",
        "usd_volume",
        "usd_marketcap",
        "media_filenames",
        "has_audio_in_video",
    ]

    for filename in os.listdir(new_json_dir):
        if filename.endswith(".json"):
            json_path = new_json_dir / filename

            new_data = dict()

            with json_path.open() as f:
                data = json.load(f)
                f.close()

            int_id = int(data["id"])
            padded_id: str = f"{int_id:05}"

            for key in sorted_keys:
                if key == "media_filenames":
                    media_filenames = data.get(key, [])
                    new_media_filenames = []
                    for media_filename in media_filenames:
                        extension = Path(media_filename).suffix
                        new_name = f"{padded_id}{extension}"
                        new_media_filenames.append(new_name)
                        new_path = media_dir / new_name
                        if not new_path.is_file():
                            print(f"Missing file: {new_name}")
                    new_data[key] = new_media_filenames
                else:
                    new_data[key] = data.get(key)

            with json_path.open("w") as f:
                json.dump(new_data, f, indent=4)


def cleanup_4(number_of_classes=10, label_name="price_class"):
    """
    1. Create price class label
    """

    all_prices = []

    for filename in os.listdir(new_json_dir):
        if filename.endswith(".json"):
            json_path = new_json_dir / filename

            with json_path.open() as f:
                data = json.load(f)
                f.close()

            eth_price = int(data.get("eth_price"))
            eth_price_decimal = int(data.get("eth_price_decimal"))
            usd_price = data.get("usd_price")
            if usd_price is None:
                usd_price = 4300
            usd_price = float(usd_price)

            actual_price = usd_price * eth_price * (10 ** (-eth_price_decimal))
            actual_price = round(actual_price, 2)
            # print(actual_price)
            all_prices.append(actual_price)

    sorted_prices = sorted(all_prices)

    percentile_prices = []
    for i in range(number_of_classes):
        percentile_idx = int(len(sorted_prices) * (i + 1) / number_of_classes) - 1
        percentile_prices.append(sorted_prices[percentile_idx])

    print(percentile_prices)

    # Traverse for the second time
    for filename in os.listdir(new_json_dir):
        if filename.endswith(".json"):
            json_path = new_json_dir / filename

            with json_path.open() as f:
                data = json.load(f)
                f.close()

            eth_price = int(data.get("eth_price"))
            eth_price_decimal = int(data.get("eth_price_decimal"))
            usd_price = data.get("usd_price")
            if usd_price is None:
                usd_price = 4300
            usd_price = float(usd_price)

            actual_price = usd_price * eth_price * (10 ** (-eth_price_decimal))
            actual_price = round(actual_price, 2)

            for idx, price in enumerate(percentile_prices):
                if actual_price <= price:
                    data[label_name] = idx
                    break

            with json_path.open("w") as f:
                json.dump(data, f, indent=4)


def get_all_unique_suffixes() -> List[str]:
    unique_suffixes = set()
    for filename in os.listdir(new_json_dir):
        if filename.endswith(".json"):
            json_path = new_json_dir / filename

            with json_path.open() as f:
                data = json.load(f)
                f.close()

            media_filenames = data.get("media_filenames", [])
            for media_filename in media_filenames:
                extension = Path(media_filename).suffix
                unique_suffixes.add(extension)
    print(unique_suffixes)
    return list(unique_suffixes)


if __name__ == "__main__":
    # cleanup_2()
    # cleanup_3()
    cleanup_4(number_of_classes=3, label_name="price_bin_3_class")
    cleanup_4(number_of_classes=20, label_name="price_bin_20_class")
    cleanup_4(number_of_classes=100, label_name="price_bin_100_class")
    ...
