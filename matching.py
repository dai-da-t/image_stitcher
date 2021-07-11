import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from homography import calc_image_size, calc_min_max_coordinate, composite_homographies


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


def match_keypoints(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    distance_threthold: float,
    ratio_threthold: float,
) -> Dict[Tuple[int, int], float]:
    """
    L2距離から2つのキーポイントのマッチングを作成
    最近傍の記述子が距離、2番目との比の二つのしきい値を満たしていればマッチとする

    Args:
        descriptors1 (np.ndarray): ベースとなる記述子集合
        descriptors2 (np.ndarray): 距離計測をする記述子集合
        distance_threthold (float): L2距離のしきい値
        ratio_threthold (float): （最近傍の距離 / 2番目との距離） のしきい値

    Returns:
        (description1のindex, description2のindex) -> 2記述子間のL2距離
        なるdict

    Note:
        Cross Checkがしやすいように上記のような返り値の形式とした。
    """
    descriptors1 = descriptors1.astype(np.int64)
    descriptors2 = descriptors2.astype(np.int64)
    matches = {}

    for i, anchor_descriptor in enumerate(descriptors1):
        # descriptors2からanchorまでの距離を計算
        distances = np.linalg.norm(descriptors2 - anchor_descriptor, axis=1)
        near_indices = np.argsort(distances)[:2]

        first_distance = distances[near_indices[0]]
        second_distance = distances[near_indices[1]]

        # 比でマッチングを絞る。SHIFT論文にて提案されている。
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
    """
    match_keypoints()を用いてn番目とn - 1番目の画像のマッチングを行う(1 <= n)

    Args:
        descriptors (np.ndarray): 記述子集合
        distance_threthold (float): L2距離のしきい値
        ratio_threthold (float): （最近傍の距離 / 2番目との距離） のしきい値
        cross_check (bool, default=false): cross_checkを行うかどうか

    Returns:
        all_matches (List[Dict]]): n番目とn - 1番目の画像のマッチング集合
                                   len(images) - 1の長さになる
                                   Dictの型詳細はmatch_keypoints()参照
    """
    all_matches: List[Dict[Tuple[int, int], float]] = []

    for i, descriptor in enumerate(descriptors):
        # 最初は計算しない
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
                # 逆側にないマッチングは除く
                if (index2, index1) not in matches_inverse:
                    tmp_matches.pop((index1, index2))

            matches = tmp_matches

        all_matches.append(matches)
    return all_matches


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
