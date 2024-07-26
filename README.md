# Florence 1k [not finished yet]

This repository contains Florence 1k, a novel dataset for monument recognition in Florence, Italy. The dataset is designed for both object detection and image retrieval tasks, featuring:

- XML annotations in PASCAL VOC format for object detection
- JSON annotations in COCO format for object detection
- a .pkl file for image retrieval

Florence 1k aims to facilitate research and development in computer vision applications focused on cultural heritage and urban landmarks.

## Overview

<!-- TODO: update -->

### Stats

<!-- TODO: update -->

- Actual number of images: `1200`
- Number of monuments: `12`

### Monument Classes

<!-- TODO: update -->

1. Cattedrale di Santa Maria del Fiore (Duomo di Firenze)
2. Battistero di San Giovanni
3. Campanile di Giotto
4. Galleria degli Uffizi
5. Loggia dei Lanzi
6. Palazzo Vecchio
7. Ponte Vecchio
8. Basilica di Santa Croce
9. Palazzo Pitti
10. Piazzale Michelangelo
11. Basilica di Santa Maria Novella
12. Basilica di San Miniato al Monte

## Dataset Structure

<!-- TODO: update -->

The dataset is organized as follows:

```
dataset/
│
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
│
└── annotations/
    ├── object_detection/
    │   ├── pascal_voc/
    │   │   ├── 0001.xml
    │   │   ├── 0002.xml
    │   │   └── ...
    │   └── coco/
    │       └── labels.json
    │
    └── image_retrieval/
        └── florence1k.pkl
```


## Data Preparation

<!-- TODO: update -->

### Augmentation

<!-- TODO: update -->
