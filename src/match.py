from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

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

    for i, descriptor in enumerate(tqdm(descriptors, desc='Matching...')):
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

        assert 4 <= len(matches), 'The number of matches is too small.'
        all_matches.append(matches)
    return all_matches
