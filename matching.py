import argparse
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


def load_images(image_paths: List[str]) -> List[np.ndarray]:
    images = []

    for image_path in image_paths:
        image: Optional[np.ndarray] = cv2.imread(image_path)
        assert image is not None, 'Failed to load "{}"'.format(image_path)

        images.append(image)
    return images


def detect_keypoints(images: List[np.ndarray]) -> Tuple[List[List[cv2.KeyPoint]], List[np.ndarray]]:
    akaze = cv2.AKAZE_create()
    keypoints: List[List[cv2.KeyPoint]] = []
    desctiptors: List[np.ndarray] = []

    for image in images:
        keypoint, desctiptor = akaze.detectAndCompute(image, mask=None)
        keypoints.append(keypoint)
        desctiptors.append(desctiptor)

    return keypoints, desctiptors


def main(args: argparse.Namespace) -> None:
    images = load_images(args.images)
    keypoints, descriptors = detect_keypoints(images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--images', help='input images', nargs='+', type=str, required=True)

    args = parser.parse_args()

    main(args)
