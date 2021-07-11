import argparse
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.homography import (
    calc_homography,
    calc_image_size,
    calc_min_max_coordinate,
    composite_homographies,
)
from src.match import match_all_images


def load_images(image_paths: List[str]) -> List[np.ndarray]:
    # 画像サイズが異なるためndarrayではなくlistに格納
    images = []

    for image_path in image_paths:
        image: Optional[np.ndarray] = cv2.imread(image_path)
        assert image is not None, 'Failed to load "{}"'.format(image_path)

        images.append(image)
    return images


def detect_keypoints(
    images: List[np.ndarray],
) -> Tuple[List[List[cv2.KeyPoint]], List[np.ndarray]]:
    """
    AKAZEを用いた特徴点検出

    Args:
        images (List[ndarray]): 画像の集合

    Returns:
        keypoints (List[List[KeyPoint]]): 各画像のKeyPoint集合
        descriptors (List[np.ndarray]): 各画像の記述子集合
    """
    akaze = cv2.AKAZE_create()
    keypoints: List[List[cv2.KeyPoint]] = []
    descriptors: List[np.ndarray] = []

    for image in images:
        keypoint, descriptor = akaze.detectAndCompute(image, mask=None)
        keypoints.append(keypoint)
        descriptors.append(descriptor)

    return keypoints, descriptors


def stitch_images(
    images: List[np.ndarray],
    homographies: np.ndarray,
    empty_image: np.ndarray,
    shift: Tuple[int, int],
    base: int = 0,
) -> np.ndarray:
    """
    画像とホモグラフィーからStitchingを行う。

    Args:
        images (List[ndarray]): 画像の集合
        homographies (ndarray): 各画像間のホモグラフィー
        empty_image (ndarray): Stitchingを行った後の画像サイズで作られた空画像
        shift (Tuple[int, int]): 負の座標値を補正するための移動量（x, y）, 最小値が0になる
        base (int): ベースとする画像

    Returns:
        stitched_image (ndarray): Stitchingを行った画像
    """
    base_image = images[base]
    base_height, base_width, _ = base_image.shape
    stitched_image = empty_image.copy()

    # shiftだけ移動させてbase画像をコピー
    stitched_image[
        shift[1] : base_height + shift[1], shift[0] : base_width + shift[0]
    ] = base_image

    for image, homography in zip(images[:base] + images[base + 1 :], homographies):
        homography = homography.reshape(3, 3)
        # 移動先から元画像の画素を取ってくるため、逆行列と移動先の最小最大を計算
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
