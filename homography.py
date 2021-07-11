from typing import List, Tuple

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
