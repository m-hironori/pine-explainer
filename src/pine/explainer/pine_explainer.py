from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from pine.entity import EntityPair, MergedSegment
from pine.explainer import AttributionScore, LimeResultPair, LimeResult
from pine.explainer.lime_explainer import make_explanation_without_separate_lr as lime_make_explanation_without_separate_lr
from pine.explainer.lime_explainer import make_explanation as lime_make_explanation
from pine.explainer.token_pair_metrics import (
    calculate_cosine_similarities_with_mean_pooling,
    determine_word_relationship,
)


@dataclass
class PairSegment:
    index_l: int
    index_r: int
    score: float


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


def select_sim_score_relationship(
    sims: np.array,
    word_pairs: List[Tuple[str, str]],
    word_relationships: List[str],
    th: float = 0.86,
) -> Tuple[int, float]:
    """以下の条件のindexとその時のsimの値を返す。
    - 同じ単語ではない。
    - 関連語（対義語か同義語（同義語ではない）） である or th以上の類似度である
    上記条件にあたはまるものがない場合は、None, -1。
    """
    is_exists_same_word = False
    idx_sorted = np.argsort(sims)[::-1]
    for idx in idx_sorted:
        w1, w2 = word_pairs[idx]
        if w1 == w2:
            is_exists_same_word = True
            continue
        if (
            word_relationships[idx] == "antonym"
            or word_relationships[idx] == "same_category"
        ):
            return idx, sims[idx], is_exists_same_word
        if sims[idx] >= th:
            return idx, sims[idx], is_exists_same_word
    return None, -1, is_exists_same_word


def make_pair_segments_core(
    entity_pair: EntityPair,
    proba_fn: Callable,
    topk: int,
    kernel: Callable,
    n_sample: int,
    random_state: int,
    fit_intercept: bool,
) -> List[PairSegment]:
    """対応したセグメントのリストを作成。score順に返す。"""
    pair_segment_list: List[PairSegment] = []

    # 通常Limeの結果
    lime_result_org = LimeResult(
        *lime_make_explanation(
            entity_pair,
            proba_fn,
            kernel=kernel,
            n_sample=n_sample,
            random_state=random_state,
            fit_intercept=fit_intercept,
        )
    )

    is_match = lime_result_org.lime_match_score > 0
    token_idxs_org_l = [attr.index for attr in lime_result_org.attributions_l]
    token_idxs_org_r = [attr.index for attr in lime_result_org.attributions_r]
    token_attrs_org_l = {x.index: x.score for x in lime_result_org.attributions_l}
    token_attrs_org_r = {x.index: x.score for x in lime_result_org.attributions_r}
    if not is_match:
        # unmatch 側の場合、スコアの正負を反転させる
        token_attrs_org_l = {idx: -score for idx, score in token_attrs_org_l.items()}
        token_attrs_org_r = {idx: -score for idx, score in token_attrs_org_r.items()}

    # 全単語ペアの類似度を計算
    word_pair_dic = {}
    for l_idx in range(entity_pair.entity_l.segment_size()):
        for r_idx in range(entity_pair.entity_r.segment_size()):
            word_pair_dic[l_idx, r_idx] = (
                entity_pair.entity_l.get_segment_label(l_idx),
                entity_pair.entity_r.get_segment_label(r_idx),
            )
    idx_pairs = list(word_pair_dic.keys())
    sims_all = calculate_cosine_similarities_with_mean_pooling(
        list(word_pair_dic.values())
    )
    word_pair_sims = {idx: sim for idx, sim in zip(idx_pairs, sims_all)}

    # 全単語ペアの関連性を判定
    word_pair_relationships = {}
    for (l_idx, r_idx), (word1, word2) in word_pair_dic.items():
        relationship = determine_word_relationship(word1, word2, hypernym_depth=1)
        word_pair_relationships[l_idx, r_idx] = relationship

    # 左側 の Tokenそれぞれに対応する右側単語を取得
    for token_idx_l in token_idxs_org_l:
        word_pairs = [word_pair_dic[token_idx_l, idx] for idx in token_idxs_org_r]
        word_relationships = [
            word_pair_relationships[token_idx_l, idx] for idx in token_idxs_org_r
        ]
        sims = np.array([word_pair_sims[token_idx_l, idx] for idx in token_idxs_org_r])
        # 右側のtokenを選択
        if is_match:
            # マッチの場合は最大のトークンを採用
            max_idx = np.argmax(sims)
            score_sim = np.max(sims)
        else:
            # アンマッチの場合、同じ単語を除外し、関連性(対義語と同位語（同義語以外）))のある単語を出力。関連性のある単語がなければ、0.86以上の単語があれば、最大のものを採用。ない場合は、採用しない。
            max_idx, score_sim, is_exists_same_word = select_sim_score_relationship(
                sims, word_pairs, word_relationships, 0.86
            )
            # 採用すべき単語がなく、かつ、同じ単語がある場合は、ペア作成しない
            if max_idx is None and is_exists_same_word:
                continue
        # token_indexとスコア
        if max_idx is None:
            token_idx_r = None
            score = token_attrs_org_l[token_idx_l]
        else:
            token_idx_r = token_idxs_org_r[max_idx]
            score = token_attrs_org_l[token_idx_l] + token_attrs_org_r[token_idx_r]

        # ペアに追加
        ## スコアが0以下の場合は、追加しない
        if score < 0:
            continue
        if not is_match:
            # unmatch側の場合、スコアの正負をもとに戻す
            score = -score
        pair_segment_list.append(
            PairSegment(
                token_idx_l,
                token_idx_r,
                score,
            )
        )
    # 右側 の Tokenそれぞれに対応する左側単語を取得
    for token_idx_r in token_idxs_org_r:
        word_pairs = [word_pair_dic[idx, token_idx_r] for idx in token_idxs_org_l]
        sims = np.array([word_pair_sims[idx, token_idx_r] for idx in token_idxs_org_l])
        word_relationships = [
            word_pair_relationships[idx, token_idx_r] for idx in token_idxs_org_l
        ]
        if is_match:
            # マッチの場合は最大のトークンを採用
            max_idx = np.argmax(sims)
            score_sim = np.max(sims)
        else:
            # アンマッチの場合、同じ単語を除外し、関連性(対義語と同位語（同義語以外）))のある単語を出力。関連性のある単語がなければ、0.9以上の単語があれば、最大のものを採用。ない場合は、採用しない。
            max_idx, score_sim, is_exists_same_word = select_sim_score_relationship(
                sims, word_pairs, word_relationships, 0.86
            )
            # 採用すべき単語がなく、かつ、同じ単語がある場合は、ペア作成しない
            if max_idx is None and is_exists_same_word:
                continue
        # token_indexとスコア
        if max_idx is None:
            token_idx_l = None
            score = token_attrs_org_r[token_idx_r]
        else:
            token_idx_l = token_idxs_org_l[max_idx]
            score = token_attrs_org_l[token_idx_l] + token_attrs_org_r[token_idx_r]

        # ペアに追加
        ## スコアが0以下の場合は、追加しない
        if score < 0:
            continue
        if not is_match:
            # unmatch側の場合、スコアの正負をもとに戻す
            score = -score
        pair_segment_list.append(
            PairSegment(
                token_idx_l,
                token_idx_r,
                score,
            )
        )

    return pair_segment_list


def make_pair_segments(
    entity_pair: EntityPair,
    proba_fn: Callable,
    topk: int,
    kernel: Callable,
    n_sample: int,
    random_state: int,
    fit_intercept: bool,
) -> List:
    """Create a list of segments."""
    pair_segment_list: List[PairSegment] = make_pair_segments_core(
        entity_pair, proba_fn, topk, kernel, n_sample, random_state, fit_intercept
    )
    # score の絶対値の大きいものからtop_n選択
    #  ただし、既に選択したtokenを含んでいる場合はSKIP
    pair_segment_list_filtered = []
    already_sel_l = set()
    already_sel_r = set()
    for pair_seg in sorted(pair_segment_list, key=lambda x: abs(x.score), reverse=True):
        if len(pair_segment_list_filtered) >= topk:
            break
        if pair_seg.index_l is not None and pair_seg.index_l in already_sel_l:
            continue
        if pair_seg.index_r is not None and pair_seg.index_r in already_sel_r:
            continue
        already_sel_l.add(pair_seg.index_l)
        already_sel_r.add(pair_seg.index_r)
        pair_segment_list_filtered.append(pair_seg)

    return pair_segment_list_filtered


def make_explanation(
    entity_pair: EntityPair,
    proba_fn: Callable,
    topk: int,
    *,
    kernel: Callable = kernel,
    n_sample: int = None,
    random_state: int = 0,
    fit_intercept: bool = True,
) -> Tuple[LimeResultPair, EntityPair]:
    """Explain the prediction of the model using PINE."""
    # STEP1: Create a list of segments
    pair_segments = make_pair_segments(
        entity_pair, proba_fn, topk, kernel, n_sample, random_state, fit_intercept
    )
    merge_segments :List[MergedSegment] = []
    for pair_seg in pair_segments:
        merge_seg = MergedSegment([], [])
        if pair_seg.index_l is not None:
            merge_seg.segment_list_in_l.append(pair_seg.index_l)
        if pair_seg.index_r is not None:
            merge_seg.segment_list_in_r.append(pair_seg.index_r)
        merge_segments.append(merge_seg)
    entity_pair_merged = entity_pair.make_entity_pair_by_merging_segment_list_only(
        merge_segments
    )
    # Explanation対象のペアがない場合は、空のLimeResultPairを返す
    if len(pair_segments) == 0:
        return LimeResultPair([], None, None, None, None), entity_pair_merged

    # STEP2: Create a list of attribution scores
    lime_result_pair = LimeResultPair(
        *lime_make_explanation_without_separate_lr(
            entity_pair_merged, proba_fn, fit_intercept=fit_intercept
        )
    )
    lime_result_pair.attributions = sorted(
        lime_result_pair.attributions, key=lambda x: abs(x.score), reverse=True
    )
    return lime_result_pair, entity_pair_merged
