import json
import os
from collections import Counter
from pathlib import Path
from pprint import pprint

data_dir = "/home/mark/CODE/NOT_MINE/project_numpie/data/raw/json"
media_dir = "/home/mark/Data/Cawin_NFT"


def load_json(filename: str) -> dict:
    filepath = Path(data_dir) / filename
    assert filepath.is_file()
    with filepath.open() as f:
        data = json.load(f)
    return data


def sanitize_string(s: str) -> str:
    for token in ["\r", "\n", "#", "\\"]:
        if token in s:
            s = s.replace(token, " ")
    return s.strip()


def combine_text_desc(i_name: str, i_desc: str, c_name: str, c_desc: str) -> str:
    i_name = sanitize_string(str(i_name))
    i_desc = sanitize_string(str(i_desc))
    c_name = sanitize_string(str(c_name))
    c_desc = sanitize_string(str(c_desc))
    return f"Name: {i_name} | Description: {i_desc} | Collection Name: {c_name} | Collection Description: {c_desc}"


def load_data():

    nft_names = []
    pic_paths = []
    texts = []
    eth_prices = []

    for name in os.listdir(data_dir):
        if name.endswith(".txt"):
            nft_name = name[:-4]
            pic_name = nft_name + "_.jpg"
            pic_path = Path(media_dir) / pic_name
            if pic_path.is_file():
                ...
            else:
                # skip for now
                continue
        else:
            continue

        data = load_json(name)
        asset_contract_data = data.get("asset_contract", {})
        payment_token_data = data.get("last_sale", {}).get("payment_token", {})

        item_name: str = data.get("name", "")
        item_desc: str = data.get("description", "")
        collection_name: str = asset_contract_data.get("name", "")
        collection_desc: str = asset_contract_data.get("description", "")

        # num_sales: int = data.get("num_sales", 0)

        # item_eth_price = actual value * 1E18
        item_eth_price: float = data.get("last_sale", {}).get("total_price")

        # NOTE: to be determined
        item_usd_price: float = payment_token_data.get("usd_price")

        if not item_eth_price or not item_usd_price:
            continue
        else:
            item_eth_price = float(item_eth_price) / 1e18
            item_usd_price = float(item_usd_price)

        text_input = combine_text_desc(
            item_name, item_desc, collection_name, collection_desc
        )

        nft_names.append(nft_name)
        pic_paths.append(str(pic_path))
        texts.append(text_input)
        eth_prices.append(item_eth_price)

    num_classes = 10
    interval = (max(eth_prices) - min(eth_prices)) / num_classes
    # print(max(eth_prices))
    # print(min(eth_prices))
    # print(interval)
    labels = [int(x // interval) for x in eth_prices]
    # for i in range(len(labels)):
    #     print(eth_prices[i], labels[i])

    return nft_names, pic_paths, texts, labels


def main():
    nft_names, pic_paths, texts, labels = load_data()
    print(len(labels))


if __name__ == "__main__":
    main()
