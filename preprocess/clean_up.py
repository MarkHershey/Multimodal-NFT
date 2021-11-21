import json
import os
from pathlib import Path

json_dir = Path("/home/mark/CODE/NOT_MINE/project_numpie/data/preprocessed/json")
out_dir = Path("/home/mark/Data/MARK_NFT/json")


def main():
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

            outpath = out_dir / f"{data['id']:05}.json"

            with outpath.open("w") as f:
                json.dump(data, f, indent=4)
                f.close()


if __name__ == "__main__":
    main()
