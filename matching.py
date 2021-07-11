import argparse
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from homography import calc_image_size, calc_min_max_coordinate, composite_homographies


def load_images(image_paths: List[str]) -> List[np.ndarray]:
    images = []

    for image_path in image_paths:
        image: Optional[np.ndarray] = cv2.imread(image_path)
        assert image is not None, 'Failed to load "{}"'.format(image_path)

        images.append(image)
    return images


def detect_keypoints(
    images: List[np.ndarray],
) -> Tuple[List[List[cv2.KeyPoint]], List[np.ndarray]]:
    akaze = cv2.AKAZE_create()
    keypoints: List[List[cv2.KeyPoint]] = []
    desctiptors: List[np.ndarray] = []

    for image in images:
        keypoint, desctiptor = akaze.detectAndCompute(image, mask=None)
        keypoints.append(keypoint)
        desctiptors.append(desctiptor)

    return keypoints, desctiptors


def match_keypoints(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    distance_threthold: float,
    ratio_threthold: float,
) -> Dict[Tuple[int, int], float]:
    descriptors1 = descriptors1.astype(np.int64)
    descriptors2 = descriptors2.astype(np.int64)
    matches = {}

    for i, anchor_descriptor in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - anchor_descriptor, axis=1)
        near_indices = np.argsort(distances)[:2]

        first_distance = distances[near_indices[0]]
        second_distance = distances[near_indices[1]]

        if (
            distance_threthold <= first_distance
            or ratio_threthold <= first_distance / second_distance
        ):
            continue

        matches[(i, near_indices[0])] = first_distance
    return matches


def match_all_images(
    descriptors: List[np.ndarray],
    distance_threthold: float,
    ratio_threthold: float,
    cross_check: bool = False,
) -> List[Dict[Tuple[int, int], float]]:
    all_matches: List[Dict[Tuple[int, int], float]] = []

    for i, descriptor in enumerate(descriptors):
        if i == 0:
            continue

        matches = match_keypoints(
            descriptor, descriptors[i - 1], distance_threthold, ratio_threthold
        )
        if cross_check:
            tmp_matches = matches.copy()
            matches_inverse = match_keypoints(
                descriptors[i - 1], descriptor, distance_threthold, ratio_threthold
            )

            for index1, index2 in matches:
                if (index2, index1) not in matches_inverse:
                    tmp_matches.pop((index1, index2))

            matches = tmp_matches

        all_matches.append(matches)
    return all_matches


def calc_homography(
    keypoints: List[cv2.KeyPoint], matches: List[Dict[Tuple[int, int], float]]
) -> np.ndarray:
    homographies = np.empty((0, 9))

    for i, match in enumerate(matches):
        coefficient = np.empty((0, 8))
        destination = np.empty((0))

        for source_index, destination_index in match:
            source_keypoint = keypoints[i + 1][source_index]
            destination_keypoint = keypoints[i][destination_index]

            destination = np.hstack((destination, destination_keypoint.pt))
            source_x, source_y = source_keypoint.pt
            destination_x, destination_y = destination_keypoint.pt
            coefficient_tmp = np.array(
                [
                    [
                        source_x,
                        source_y,
                        1,
                        0,
                        0,
                        0,
                        -source_x * destination_x,
                        -source_y * destination_x,
                    ],
                    [
                        0,
                        0,
                        0,
                        source_x,
                        source_y,
                        1,
                        -source_x * destination_y,
                        -source_y * destination_y,
                    ],
                ]
            )
            coefficient = np.vstack((coefficient, coefficient_tmp))

        homography = np.linalg.pinv(coefficient) @ destination
        homography = np.hstack((homography, 1))
        homographies = np.vstack((homographies, homography))

    return homographies


def stitch_images(
    images: List[np.ndarray],
    homographies: np.ndarray,
    empty_image: np.ndarray,
    shift: Tuple[int, int],
    base: int = 0,
):
    base_image = images[base]
    base_height, base_width, _ = base_image.shape
    stitched_image = empty_image.copy()

    stitched_image[
        shift[1] : base_height + shift[1], shift[0] : base_width + shift[0]
    ] = base_image

    for image, homography in zip(images[:base] + images[base + 1 :], homographies):
        homography = homography.reshape(3, 3)
        inv_homography = np.linalg.inv(homography)
        min_x, min_y, max_x, max_y = calc_min_max_coordinate(image, homography)

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                source_coord = inv_homography @ [x, y, 1]
                source_x, source_y = source_coord[:-1].clip(0).astype(np.int64)
                assert 0 <= source_x and 0 <= source_y

                stitched_image[y + shift[1], x + shift[0]] = image[
                    source_y, source_x, :
                ]

    return stitched_image.clip(0, 255).astype(np.uint8)


def main(args: argparse.Namespace) -> None:
    images = load_images(args.images)
    keypoints, descriptors = detect_keypoints(images)
    matches = match_all_images(
        descriptors, args.distance_threthold, args.ratio_threthold, args.cross_check
    )
    homographies = calc_homography(keypoints, matches)

    composited_homographies = composite_homographies(homographies)
    width, height, shift = calc_image_size(images, composited_homographies)
    empty_image = np.zeros((height, width, 3))

    stitched_image = stitch_images(images, composited_homographies, empty_image, shift)
    cv2.imshow("img_stitched", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--images", help="input images", nargs="+", type=str, required=True
    )

    parser.add_argument(
        "-d",
        "--distance_threthold",
        help="threshold of L2 distance for matching",
        type=float,
        default=10,
    )
    parser.add_argument(
        "-r",
        "--ratio_threthold",
        help="threshold of the ratio of the first to the second in matching",
        type=float,
        default=0.8,
    )
    parser.add_argument("-c", "--cross_check", action="store_true")

    args = parser.parse_args()

    main(args)
