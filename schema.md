# Annotation Schema

-   One JSON per NFT
-   Remove all non-ASCII characters for name and descriptions.
-   Replace change line "\n" with space " " in descriptions.
-   Put all media filenames in the list.
-   "eth_price" is the last sale total price, unit: eth times 10 to the power of negative "eth_price_decimal".
-   "usd_price" is the equivalent price to "eth_price" in USD, conversion using market data on that day.
-   If possible, Just need to indicate if there is audio in the mp4 file.
    -   If no mp4 file, put false.
    -   If the mp4 file has no sound, put false.
    -   Put true only if there is a mp4 file and it has sound track within it.

```json
{
    "id": 6382,
    "name": "Go NFT - Namewee #057",
    "description": "This is the only song dedicated to NFT in the world.",
    "collection_name": "NAMEWEE4896 Collection",
    "collection_description": "",
    "transaction_time": "2021-11-07T06:59:21",
    "eth_price": "1000000000000000000",
    "eth_price_decimal": 18,
    "usd_price": null,
    "usd_volume": null,
    "usd_marketcap": null,
    "media_filenames": [
        "06382.mp3",
        "06382.jpg"
    ],
    "has_audio_in_video": false,
    "price_class": 10
}%
```
