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

Experiment Batch 1

| Experiment         | Test Accuracy | Remark                       | Best Epoch |
| ------------------ | ------------- | ---------------------------- | ---------- |
| `base`             | 0.4400        | B=64, resnet34               | 14         |
| `resnet50`         | 0.4385        | B=64, resnet50               | 8          |
| `pure_text`        | 0.4400        | B=64, resnet34               | 14         |
| `base_filter`      | 0.4689        | B=64, resnet34, 2020 onwards | 17         |
| `resnet50_filter`  | 0.4705        | B=64, resnet50, 2020 onwards | 28         |
| `pure_text_filter` | 0.4689        | B=64, resnet34, 2020 onwards | 17         |

---

Experiment Batch 2

| Experiment | Test Accuracy | Remark | Best Epoch |
| ---------- | ------------- | ------ | ---------- |
| `exp_b3`   | ------------- | ------ | ---------- |
| `exp_b10`  | ------------- | ------ | ---------- |
| `exp_b20`  | ------------- | ------ | ---------- |
| `exp_b100` | ------------- | ------ | ---------- |
