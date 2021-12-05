# Multi-modal NFT Price Prediction

SUTD 50.038 Computational Data Science (Fall 2021) - **Team Numpie**

![](https://img.shields.io/badge/Team-Numpie-green?style=for-the-badge)
![](https://img.shields.io/github/license/MarkHershey/Multimodal-NFT?style=for-the-badge)

## Abstract

The recent rise in cryptocurrency and blockchain technologies has led to a surge in interest in Non-Fungible Tokens (NFTs), which are uniquely identified digital assets that represent virtual objects such as art, music, and in-game characters. Public interest in NFT has exploded due to skyrocketing NFT prices, however, the overall structure and value of an NFT is still a mystery, largely because the diversity of virtual assets makes it difficult to model its price across different mediums and domains.

To understand NFT prices better across domains, we introduce a new dataset that consists of 10454 diverse multi-media NFTs across different NFT categories, consisting of music-, image-, text-, video-based NFT media files and labeled with its transactional data, all collected from OpenSea. This is done in order to conduct price estimation across different forms of NFTs. We also propose a multi-modal deep learning-based network that is capable of predicting prices universally regardless of the forms of the NFT assets.

## NFT Dataset

-   Dataset consists of 10454 NFTs, each of which has a corresponding JSON file and one or more media files in the format of `.jpg`, `.gif`, `.mp4`, and/or `.mp3`.
-   All JSON files is contained here [`data/NFT-JSON.zip`](data/NFT-JSON.zip), please extract it to `data/json` directory.
-   All media files can be downloaded [[here]](https://drive.google.com/file/d/1dS4KSPYtlwGPN7uI6EQE157L5wDozX8C/view?usp=sharing). Please download it and extract the files to `data/media` directory.

Example JSON Label:

```json
{
    "id": 1234,
    "name": "CryptoPunk #1284",
    "description": null,
    "collection_name": "CryptoPunks",
    "collection_description": "CryptoPunks launched as a fixed set of 10,000 items in mid-2017 and became one of the inspirations for the ERC-721 standard. They have been featured in places like The New York Times, Christies of London, Art|Basel Miami, and The PBS NewsHour.",
    "transaction_time": "2017-07-04T12:07:31",
    "eth_price": "250000000000000000",
    "eth_price_decimal": 18,
    "usd_price": 273.303009,
    "usd_volume": 687691008.0,
    "usd_marketcap": 25425860777.0,
    "media_filenames": ["01234.jpg"],
    "has_audio_in_video": false,
    "price_class": 0,
    "price_bin_3_class": 0,
    "price_bin_20_class": 1,
    "price_bin_100_class": 7
}
```

## Model Training

### Dependencies

-   Python 3.6 or above
-   [PyTorch 1.10](https://pytorch.org/)
-   [`requirements.txt`](requirements.txt)

1. Create a virtual environment

```bash
python3 -m venv venv && source venv/bin/activate
```

2. Install dependencies from [`requirements.txt`](requirements.txt) first

```bash
pip install -r requirements.txt
```

3. Install the PyTorch version that [suits your machine](https://pytorch.org/get-started/locally/), use GPU version whenever possible.

```bash
pip3 install torch torchvision torchaudio
```

### Preprocessing

To extract the features from the NFT media files, we use the following preprocessing pipeline:

-   [preprocess/process_audio.py](preprocess/process_audio.py)
-   [preprocess/process_image.py](preprocess/process_image.py)
-   [preprocess/process_motion.py](preprocess/process_motion.py)
-   [preprocess/process_textual.py](preprocess/process_textual.py)

Example command to extract features from audio files:

```bash
python preprocess/process_audio.py --json_dir data/json --media_dir data/media --features_dim 3600
```

Save all extracted features binary artifacts to `data/` directory.

### Training

Default training parameters are defined in the [`config.py`](config.py) file. You can define your own experiment by either directly modify the [`config.py`](config.py) file or use a config file like this [`cfgs/template.json`](cfgs/template.json). Any parameters defined in the config file will overwrite the default parameters.

To run the training code with default configs:

```bash
python train.py
```

To run the training code with your own configs:

```bash
python train.py --cfg cfgs/template.json
```

-   Training logs, model weights for the best-performing model, and the best testing predictions outputs will be saved in the `results` directory by default.
-   [run_exps.py](run_exps.py) helps you run a list of experiments in serial based on given a list of config files.

## Citation

If you use this code or dataset, please cite as follows:

```bibtex
@misc{Huang_MMNFT_2021,
    author  = {Huang, He and Chan, Cawin and Poh, Princeton},
    month   = {12},
    title   = {{Multi-Modal NFT Price Prediction}},
    url     = {https://github.com/MarkHershey/Multimodal-NFT},
    version = {0.0.1},
    year    = {2021}
}
```
