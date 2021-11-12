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
    "RID": "1234",
    "name": "NFT Name",
    "description": "NFT Description",
    "collection_name": "Collection Name",
    "collection_description": "Collection Description",
    "eth_price": 9000000000000000000,
    "eth_price_decimal": 18,
    "usd_price": 1200.89,
    "transaction_time": "2021-07-23T10:16:28",
    "media_filenames": ["1234.mp4", "1234.mp3"],
    "has_audio_in_video": false
}
```
