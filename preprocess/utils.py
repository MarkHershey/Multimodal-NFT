import json
import os
import re
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup

data_dir = "/home/mark/Data/NFT_Dataset/json"
media_dir = "/home/mark/Data/NFT_Dataset/media"


def clean_text(text: str) -> str:
    REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
    BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
    # HTML decoding
    text = BeautifulSoup(text, "html.parser").text
    # make lowercase
    text = text.lower()
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(" ", text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub("", text)
    return text.strip()


def sanitize_string(s: str) -> str:
    for token in ["\r", "\n", "#", "\\"]:
        if token in s:
            s = s.replace(token, " ")
    return s.strip()


def combine_text_desc(
    i_name: str, i_desc: str, c_name: str, c_desc: str, with_header: bool = False
) -> str:
    i_name = sanitize_string(str(i_name))
    i_desc = sanitize_string(str(i_desc))
    c_name = sanitize_string(str(c_name))
    c_desc = sanitize_string(str(c_desc))
    if with_header:
        return f"Name: {i_name} | Description: {i_desc} | Collection Name: {c_name} | Collection Description: {c_desc}"
    else:
        return f"{i_name} {i_desc} {c_name} {c_desc}"


def load_text_data(data_dir: str = data_dir) -> Tuple[list, list]:
    ids: List[int] = []
    texts: List[str] = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            filepath = Path(data_dir) / filename
            with filepath.open() as f:
                json_data = json.load(f)
                f.close()

            _id = json_data.get("id", "")
            ids.append(_id)
            i_name = json_data.get("name", "")
            i_desc = json_data.get("description", "")
            c_name = json_data.get("collection_name", "")
            c_desc = json_data.get("collection_description", "")
            # combine name and descriptions into one string
            text = combine_text_desc(i_name, i_desc, c_name, c_desc, with_header=False)
            # clean up text
            text = clean_text(text)
            texts.append(text)

    return ids, texts


def main():
    ...


if __name__ == "__main__":
    main()
