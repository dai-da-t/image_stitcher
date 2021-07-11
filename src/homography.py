from typing import List, Tuple

import cv2
import numpy as np


def calc_min_max_coordinate(
    image: np.ndarray, homography: np.ndarray
) -> Tuple[int, int, int, int]:
    height, weight, _ = image.shape
    min_x, min_y = np.inf, np.inf
    max_x, max_y = 0, 0

    corners = np.array([[0, 0, 1], [0, height, 1], [weight, 0, 1], [weight, height, 1]])
    for corner in corners:
        warped_corner = homography.reshape(3, 3) @ corner
        x = warped_corner[0]
        y = warped_corner[1]

        min_x = min(x, min_x)
        min_y = min(y, min_y)
        max_x = max(x, max_x)
        max_y = max(y, max_y)

    min_x, min_y, max_x, max_y = map(int, (min_x, min_y, max_x, max_y))
    return min_x, min_y, max_x, max_y


def calc_image_size(
    images: List[np.ndarray], homographies: np.ndarray
) -> Tuple[int, int, Tuple[int, int]]:
    min_x, min_y = 0, 0
    max_y, max_x, _ = images[0].shape

    for image, homography in zip(images[1:], homographies):
        min_x_tmp, min_y_tmp, max_x_tmp, max_y_tmp = calc_min_max_coordinate(
            image, homography
        )

        min_x = min(min_x_tmp, min_x)
        min_y = min(min_y_tmp, min_y)
        max_x = max(max_x_tmp, max_x)
        max_y = max(max_y_tmp, max_y)

    width = max_x - min_x
    height = max_y - min_y
    shift = (-min_x, -min_y)

    return width, height, shift


def composite_homographies(homographies: np.ndarray) -> np.ndarray:
    homographies_reverse = homographies[::-1]
    composited_homographies = np.empty((0, 9))

    for i, composited_homography in enumerate(homographies):
        composited_homography = composited_homography.reshape((3, 3))
        for homography in homographies_reverse[len(homographies) - i :]:
            homography = homography.reshape((3, 3))
            composited_homography = homography @ composited_homography

        composited_homography = composited_homography.reshape(1, 9)

        composited_homographies = np.vstack(
            (composited_homographies, composited_homography)
        )

    return composited_homographies


def calc_homography(
    keypoints: List[cv2.KeyPoint], matches: List[Dict[Tuple[int, int], float]]
) -> np.ndarray:
    """
    マッチングから特徴点の座標ペアを作成し、ホモグラフィーを計算

    Args:
        keypoints (List[List[KeyPoint]]): 各画像のKeyPoint集合
        matches (List[Dict]): 各画像のマッチング集合

    Returns:
        homographies (ndarray): 各画像のホモグラフィー集合
                                shape: (画像数, 9)
    """
    homographies = np.empty((0, 9))

    for i, match in enumerate(matches):
        # Ah = bの方程式を作成
        # A : coefficient, b: destination
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

        # 一般逆行列を用いてホモグラフィーを計算
        homography = np.linalg.pinv(coefficient) @ destination
        homography = np.hstack((homography, 1))
        homographies = np.vstack((homographies, homography))

    return homographies
