from typing import Callable, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from lime import lime_base
from pine.entity import EntityPair
from pine.explainer import AttributionScore


def mask_segments(
    entity_pair: EntityPair,
    zs: np.ndarray,
    mask_token_str: str = None,
) -> np.ndarray:
    """Entity pair と マスクリストを入力に、マスクされた Entity Pairリストを出力する

    Args:
        entity_pair (EntityPair): エンテイペア
        zs (np.ndarray): マスクセグメントを0にしたデータ列。zs.shape = (data_len, segment_len).
        mask_token_str (str, optional): マスク位置に挿入する文字列. Defaults to None.

    Returns:
        numpy.ndarray: masked_entity_pair_list: entity_pairのリスト。len(entity_pair) = zs.shape[0]
    """
    assert zs.shape[1] == entity_pair.segment_size()

    masked_entity_pair_list: List[EntityPair] = []
    for z in zs:
        index_list = np.where(z == 0)[0]
        masked_entity_pair = entity_pair.make_entity_pair_by_deleting_segments(
            index_list, mask_token_str
        )
        masked_entity_pair_list.append(masked_entity_pair)
    return masked_entity_pair_list


def generate_batch(lst, batch_size):
    """Yields batch of specified size"""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def make_data_with_max_mask_until_num_data(
    entity_pair: EntityPair,
    max_mask: int,
    num_data: int,
    proba_fn: Callable,
    mask_token_str: str = None,
    add_all_mask_data: bool = False,
    random_state: int = 0,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ターゲットの周辺データを作成

    マスク数max_maskまでのデータをランダムで作成。num_data個のデータを作成。
    ただし、マスクなしデータを先頭に必ず追加。add_all_mask_dataがTrueの時、2番目に全マスクのデータも追加。

    * entity_pair :元のデータペア
    * max_mask (int) : 最大マスク数
    * num_data (int): 作成するデータ数
    * proba_fn: entity_pairのDataframe表現を入力に、スコアを出力する関数(score[idx] <- proba(entity_pair[idx])
    * mask_token_str : Maskする位置に採用するToken(None以外だったら用いる)
    * add_all_mask_data : 全てmaskしたデータを追加するか
    * random_state: 乱数のシード

    Returns:
    * zs (np.ndarray): データ(z空間) (zs.shape=(num_data, len(entity_pairのz空間)))
    * scores(np.ndarray): 作成データに対応するスコア (scores.shape=(num_data,))
    * distances(np.ndarray): 作成データと元データの距離 (distances.shape=(num_data,))
    """
    if random_state is None:
        random_state = np.random.default_rng()
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    if batch_size is None:
        batch_size = num_data

    # maskするindex番号をnum_data分作成
    # 最初に、全部ありデータ、全マスクデータを追加
    mask_indices = []
    min_mask = 0
    num_features = entity_pair.segment_size()
    if num_data is None:
        # num_data = max(min(30 * num_features, 3000), 500)  # from lemon
        num_data = max(min(30 * 2 * num_features, 3000 * 2), 500 * 2)  # from lemon * 2
    if max_mask is None:
        # from lemon
        max_mask = min(max(5, num_features // 5), num_features)
    # 全部ありデータ
    mask_indices.append([])
    ## 全部なしデータ
    if add_all_mask_data:
        mask_indices.append(list(range(num_features)))
    # のこり必要なデータ数
    num_data_remain = num_data - len(mask_indices)
    for _ in range(num_data_remain):
        mask_count = (
            random_state.integers(min_mask, max_mask + 1) if num_features > 0 else 0
        )
        mask_indices.append(
            random_state.choice(num_features, replace=False, size=mask_count)
        )

    # 周辺データ作成
    zs = np.empty((0, num_features))
    scores = np.empty((0, 1))
    distances = np.empty((0,))
    # batch_size 毎にデータ生成
    for i, mask_indices_batch in enumerate(generate_batch(mask_indices, batch_size)):
        # print(f"\nMaking data {i} batch..", end="")
        zs_batched = np.ones((len(mask_indices_batch), num_features), dtype=np.int8)
        for j, idxs in enumerate(mask_indices_batch):
            if len(idxs) == 0:  # マスクしないデータの場合
                continue
            zs_batched[j, idxs] = 0
        masked_entity_pair_list_batched = mask_segments(
            entity_pair, zs_batched, mask_token_str
        )

        # スコアを計算
        scores_batched = proba_fn(masked_entity_pair_list_batched)

        # z空間での距離を計算
        # from lemon ハミング距離(=マスクした数)
        distances_batched = np.zeros(len(mask_indices_batch))
        for j, idxs in enumerate(mask_indices_batch):
            if len(idxs) == 0:  # マスクしないデータの場合
                continue
            distances_batched[j] = min(len(idxs), max_mask)

        zs = np.vstack((zs, zs_batched))
        scores = np.vstack((scores, scores_batched))
        distances = np.hstack((distances, distances_batched))

    distances = distances / max_mask
    return zs, scores, distances


def kernel(d: np.ndarray) -> float:
    """カーネル関数(from lemon)

        kernel func (distances):
            exp^(-2 * d)

    Args:
        d (int): 距離（ハミング距離=マスクしたデータの長さ）

    Returns:
        float: 重み
    """
    return np.exp(-2 * d)


def make_explanation(
    entity_pair: EntityPair,
    proba_fn: Callable,
    kernel: Callable = kernel,
    n_sample: int = None,
    random_state: int = 0,
    fit_intercept: bool = True,
) -> Tuple[
    List[AttributionScore], List[AttributionScore], float, float, float, float
]:
    """LIMEの結果を作成する

    Args:
    * entity_l : 比較対象商品左
    * entity_r : 比較対象商品右
    * proba_fn: entity_pairのリストを入力にスコアを出力する関数(score[idx] <- proba(entity_pair[idx]))
    * kernel(function) : Limeのカーネル関数(距離を引数),
    * n_sample(int) : LIME実行時にサンプリングするデータ数,
    * random_state(int)
    * fit_intercept(bool) : limeの線形Fit時に切片を含めるか

    Returns:
    * attribution_scores_l(List) : 左商品の単語idx, スコアのペアのリスト,
    * attribution_scores_r(List) : 右商品の単語idx, スコアのペアのリスト,
    * right : マスクしない時のオリジナルモデルのスコア,
    * intercept: limeの線形モデルの切片,
    * prediction_score: limeの線形モデルのロス,
    * local_pred : limeの線形モデルのマスクしない時のスコア,
    """
    (
        neighbor_data,
        neighbor_scores,
        neighbor_dist,
    ) = make_data_with_max_mask_until_num_data(
        entity_pair,
        None,
        n_sample,
        proba_fn,
        mask_token_str=None,
        add_all_mask_data=False,
        random_state=random_state,
    )

    # 一つ目のデータはマスクなしデータなので、ここから実際のスコアを抽出
    right = neighbor_scores[0][0]

    # lime で attribution scoreを計算
    lime = lime_base.LimeBase(kernel, verbose=False, random_state=0)
    model_regressor = LinearRegression(fit_intercept=fit_intercept)
    (
        intercept,
        feature_scores,
        prediction_score,
        local_pred,
    ) = lime.explain_instance_with_data(
        neighbor_data,
        neighbor_scores,
        neighbor_dist,
        0,
        num_features=None,
        feature_selection="none",
        model_regressor=model_regressor,
    )

    # item_lとitem_rに分ける
    feature_scores_l = []
    feature_scores_r = []
    for idx, feature_score in feature_scores:
        if idx < entity_pair.entity_l.segment_size():
            feature_scores_l.append(AttributionScore(idx, feature_score))
        else:
            idx_in_r = idx - entity_pair.entity_l.segment_size()
            feature_scores_r.append(AttributionScore(idx_in_r, feature_score))

    return (
        feature_scores_l,
        feature_scores_r,
        right,
        intercept,
        prediction_score,
        local_pred[0],
    )


def make_explanation_without_separate_lr(
    entity_pair: EntityPair,
    proba_fn: Callable,
    kernel: Callable = kernel,
    n_sample: int = None,
    random_state: int = 0,
    fit_intercept: bool = True,
) -> Tuple[
    List[AttributionScore], float, float, float, float
]:
    """LIMEの結果を作成する

    Args:
    * entity_pair : 比較対象商品ペア
    * proba_fn: entity_pairのリストを入力にスコアを出力する関数(score[idx] <- proba(entity_pair[idx]))
    * kernel(function) : Limeのカーネル関数(距離を引数),
    * n_sample(int) : LIME実行時にサンプリングするデータ数,
    * random_state(int)
    * fit_intercept(bool) : limeの線形Fit時に切片を含めるか

    Returns:
    * attribution_scores(List) : 商品ペアの単語idx, スコアのペアのリスト,
    * right : マスクしない時のオリジナルモデルのスコア,
    * intercept: limeの線形モデルの切片,
    * prediction_score: limeの線形モデルのロス,
    * local_pred : limeの線形モデルのマスクしない時のスコア,
    """
    (
        neighbor_data,
        neighbor_scores,
        neighbor_dist,
    ) = make_data_with_max_mask_until_num_data(
        entity_pair,
        None,
        n_sample,
        proba_fn,
        mask_token_str=None,
        add_all_mask_data=False,
        random_state=random_state,
    )

    # 一つ目のデータはマスクなしデータなので、ここから実際のスコアを抽出
    right = neighbor_scores[0][0]

    # lime で attribution scoreを計算
    lime = lime_base.LimeBase(kernel, verbose=False, random_state=0)
    model_regressor = LinearRegression(fit_intercept=fit_intercept)
    (
        intercept,
        feature_scores,
        prediction_score,
        local_pred,
    ) = lime.explain_instance_with_data(
        neighbor_data,
        neighbor_scores,
        neighbor_dist,
        0,
        num_features=None,
        feature_selection="none",
        model_regressor=model_regressor,
    )

    attribution_scores = []
    for idx, feature_score in feature_scores:
        attribution_scores.append(AttributionScore(idx, feature_score))

    return (
        attribution_scores,
        right,
        intercept,
        prediction_score,
        local_pred[0],
    )
