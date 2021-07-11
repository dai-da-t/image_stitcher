# Image Stitcher
---

## Requirements
- OpenCV
- numpy
- tqdm

```
$ poetry install
```

## Usage
```
$ python stitch.py -i IMAGES -o OUTPUT [-d DISTANCE_THRETHOLD] [-r RATIO_THRETHOLD] [-a {AKAZE,SIFT}] [-c]
```