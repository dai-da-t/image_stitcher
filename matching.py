import argparse
from typing import Dict, List, Optional, Tuple, Union

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


def match_keypoints(descriptors1: np.ndarray, descriptors2: np.ndarray, distance_threthold: float, ratio_threthold: float) -> Dict[Tuple[int, int], float]:
    descriptors1 = descriptors1.astype(np.int64)
    descriptors2 = descriptors2.astype(np.int64)
    matches = {}

    for i, anchor_descriptor in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - anchor_descriptor, axis=1)
        near_indices = np.argsort(distances)[:2]

        first_distance = distances[near_indices[0]]
        second_distance = distances[near_indices[1]]

        if (distance_threthold <= first_distance or
            ratio_threthold <= first_distance / second_distance):
            continue

        matches[(i, near_indices[0])] = first_distance
    return matches


def match_all_images(descriptors: List[np.ndarray], distance_threthold: float, ratio_threthold: float, cross_check: bool = False):
    all_matches = []

    for i, descriptor in enumerate(descriptors):
        if i == 0:
            continue

        matches = match_keypoints(descriptor, descriptors[i-1], distance_threthold, ratio_threthold)
        if cross_check:
            tmp_matches = matches.copy()
            matches_inverse = match_keypoints(descriptors[i-1], descriptor, distance_threthold, ratio_threthold)

            for index1, index2 in matches:
                if (index2, index1) not in matches_inverse:
                    tmp_matches.pop((index1, index2))

            matches = tmp_matches

        all_matches.append(matches)
    return all_matches


def main(args: argparse.Namespace) -> None:
    images = load_images(args.images)
    keypoints, descriptors = detect_keypoints(images)
    matches = match_all_images(descriptors, args.distance_threthold, args.ratio_threthold, args.cross_check)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--images', help='input images', nargs='+', type=str, required=True)

    parser.add_argument('-d', '--distance_threthold', help='threshold of L2 distance for matching', type=float, default=10)
    parser.add_argument('-r', '--ratio_threthold', help='threshold of the ratio of the first to the second in matching', type=float, default=0.8)
    parser.add_argument('-c', '--cross_check', action='store_true')

    args = parser.parse_args()

    main(args)
