# Multimodal-NFT

-   Multi-Modal
    -   Images
        -   `image/jpeg`
        -   `image/png`
        -   `image/svg+xml`
    -   Audio
    -   Video (with audio)
    -   Video (without audio) / Images sequence
        -   `image/gif`
    -   Text Descriptions
        -   `text/plain`

## Feature Extraction

-   Text
    -   GloVe-300
-   Image
    -   Resnet
-   Image Sequence
    -   Resnet18
-   Audio
    -   TODO

## Training Result

| Experiment  | Test Accuracy | Remark             | Best Epoch |
| ----------- | ------------- | ------------------ | ---------- |
| `exp18`     | 0.4443        | freeze glove, B=16 | 9          |
| `exp18-64`  | 0.4495        | freeze glove, B=64 | 13         |
| `exp18-64L` | 0.4452        | train glove, B=64  | 9          |
| `exp34`     | 0.4443        | freeze glove, B=16 | 8          |
| `exp34`     | 0.4443        | freeze glove, B=16 | 8          |
| `exp50`     | 0.445         | freeze glove, B=16 | 16         |
| `exp50L`    | 0.4371        | train glove, B=16  | 4          |
| `exp101`    | 0.4452        | freeze glove, B=16 | 16         |

---

| Experiment | Test Accuracy | Remark | Best Epoch |
| ---------- | ------------- | ------ | ---------- |
