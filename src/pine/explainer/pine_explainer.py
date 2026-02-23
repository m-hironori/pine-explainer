from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from pine.entity import EntityPair, MergedSegment
from pine.explainer import AttributionScore, LimeResultPair, LimeResult
from pine.explainer.lime_explainer import (
    make_explanation_without_separate_lr as lime_make_explanation_without_separate_lr,
)
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


def same_sign(a: float, b: float) -> bool:
    """aとbが同じ符号かどうか"""
    return (a >= 0 and b >= 0) or (a < 0 and b < 0)


def select_most_related_token(
    sims: np.array,
    word_relationships: List[str],
    word_pairs: List[Tuple[str, str]],
    th: float = 0.86,
) -> Tuple[int, float]:
    """関連性のある単語を選択する。関連性のある単語がない場合は、類似度が最も高い単語を選択する。"""
    # 関連性のある単語の中で、類似度が最も高い単語を選択する
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


def extract_correlated_token_pairs(
    entity_pair: EntityPair,
    score_fn: Callable,
    topk: int,
    kernel: Callable,
    n_sample: int,
    random_state: int,
    fit_intercept: bool,
    batch_size: int = 512,
) -> List[PairSegment]:
    """対応したセグメントのリストを作成。score順に返す。"""
    pair_segment_list_cand: List[PairSegment] = []

    lime_result_org = LimeResult(
        *lime_make_explanation(
            entity_pair,
            score_fn,
            kernel=kernel,
            n_sample=n_sample,
            random_state=random_state,
            fit_intercept=fit_intercept,
        )
    )
    token_l_attr_score = {
        attr.index: attr.score for attr in lime_result_org.attributions_l
    }
    token_r_attr_score = {
        attr.index: attr.score for attr in lime_result_org.attributions_r
    }

    # Model’s match score for the record pair
    match_score = lime_result_org.match_score

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
        [word_pair_dic[idx_pair] for idx_pair in idx_pairs], batch_size=batch_size
    )
    word_pair_sims = {idx: sim for idx, sim in zip(idx_pairs, sims_all)}

    # 全単語ペアの関連性を判定
    word_pair_relationships = {}
    for (l_idx, r_idx), (word1, word2) in word_pair_dic.items():
        relationship = determine_word_relationship(word1, word2, hypernym_depth=1)
        word_pair_relationships[l_idx, r_idx] = relationship

    # Create candidate token pairs
    for token_idx_l in range(entity_pair.entity_l.segment_size()):
        token_idxs_r = range(entity_pair.entity_r.segment_size())
        sims = [word_pair_sims[(token_idx_l, idx)] for idx in token_idxs_r]
        relations = [
            word_pair_relationships[(token_idx_l, idx)] for idx in token_idxs_r
        ]
        word_pairs = [word_pair_dic[token_idx_l, idx] for idx in token_idxs_r]
        if same_sign(match_score, token_l_attr_score[token_idx_l]):
            if match_score > 0:
                # SelectMostSimilarToken
                token_idx_r = np.argmax(sims)
            else:
                token_idx_r, score_sim, is_exists_same_word = select_most_related_token(
                    sims, relations, word_pairs, th=0.86
                )
                # 採用すべき単語がなく、かつ、同じ単語がある場合は、ペア作成しない
                if token_idx_r is None and is_exists_same_word:
                    continue
            pair_segment_list_cand.append(PairSegment(token_idx_l, token_idx_r, None))
    for token_idx_r in range(entity_pair.entity_r.segment_size()):
        token_idxs_l = range(entity_pair.entity_l.segment_size())
        sims = [word_pair_sims[(idx, token_idx_r)] for idx in token_idxs_l]
        relations = [
            word_pair_relationships[(idx, token_idx_r)] for idx in token_idxs_l
        ]
        word_pairs = [word_pair_dic[idx, token_idx_r] for idx in token_idxs_l]
        if same_sign(match_score, token_r_attr_score[token_idx_r]):
            if match_score > 0:
                # SelectMostSimilarToken
                token_idx_l = np.argmax(sims)
            else:
                token_idx_l, score_sim, is_exists_same_word = select_most_related_token(
                    sims, relations, word_pairs, th=0.86
                )
                # 採用すべき単語がなく、かつ、同じ単語がある場合は、ペア作成しない
                if token_idx_l is None and is_exists_same_word:
                    continue
            pair_segment_list_cand.append(PairSegment(token_idx_l, token_idx_r, None))

    #  Calculate pseudo attribution scores
    for pair_seg in pair_segment_list_cand:
        score_l = (
            token_l_attr_score[pair_seg.index_l] if pair_seg.index_l is not None else 0
        )
        score_r = (
            token_r_attr_score[pair_seg.index_r] if pair_seg.index_r is not None else 0
        )
        pair_seg.score = score_l + score_r

    # Greedy selection of top K pairs
    if match_score >= 0:
        pair_segment_list_cand = sorted(
            pair_segment_list_cand, key=lambda x: x.score, reverse=True
        )
    else:
        pair_segment_list_cand = sorted(pair_segment_list_cand, key=lambda x: x.score)
    pair_segment_list_filtered = []
    already_sel_l = set()
    already_sel_r = set()
    for pair_seg in pair_segment_list_cand:
        if len(pair_segment_list_filtered) >= topk:
            break
        if pair_seg.index_l is not None and pair_seg.index_l in already_sel_l:
            continue
        if pair_seg.index_r is not None and pair_seg.index_r in already_sel_r:
            continue
        if same_sign(match_score, pair_seg.score):
            already_sel_l.add(pair_seg.index_l)
            already_sel_r.add(pair_seg.index_r)
            pair_segment_list_filtered.append(pair_seg)

    return pair_segment_list_filtered


def extract_correlated_token_pair_cossim(
    entity_pair: EntityPair,
    score_fn: Callable,
    topk: int,
    kernel: Callable,
    n_sample: int,
    random_state: int,
    fit_intercept: bool,
    batch_size: int = 512,
) -> List[PairSegment]:
    """対応したセグメントのリストを作成。score順に返す。"""
    pair_segment_list_cand: List[PairSegment] = []

    # Model’s match score for the record pair
    match_score = score_fn([entity_pair])[0]

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
        [word_pair_dic[idx_pair] for idx_pair in idx_pairs], batch_size=batch_size
    )
    word_pair_sims = {idx: sim for idx, sim in zip(idx_pairs, sims_all)}

    # 全単語ペアの関連性を判定
    word_pair_relationships = {}
    for (l_idx, r_idx), (word1, word2) in word_pair_dic.items():
        relationship = determine_word_relationship(word1, word2, hypernym_depth=1)
        word_pair_relationships[l_idx, r_idx] = relationship

    # Create candidate token pairs
    for token_idx_l in range(entity_pair.entity_l.segment_size()):
        token_idxs_r = range(entity_pair.entity_r.segment_size())
        sims = [word_pair_sims[(token_idx_l, idx)] for idx in token_idxs_r]
        relations = [
            word_pair_relationships[(token_idx_l, idx)] for idx in token_idxs_r
        ]
        word_pairs = [word_pair_dic[token_idx_l, idx] for idx in token_idxs_r]
        if match_score > 0:
            # SelectMostSimilarToken
            token_idx_r = np.argmax(sims)
        else:
            token_idx_r, score_sim, is_exists_same_word = select_most_related_token(
                sims, relations, word_pairs, th=0.86
            )
            # 採用すべき単語がなく、かつ、同じ単語がある場合は、ペア作成しない
            if token_idx_r is None and is_exists_same_word:
                continue
        pair_segment_list_cand.append(PairSegment(token_idx_l, token_idx_r, None))
    for token_idx_r in range(entity_pair.entity_r.segment_size()):
        token_idxs_l = range(entity_pair.entity_l.segment_size())
        sims = [word_pair_sims[(idx, token_idx_r)] for idx in token_idxs_l]
        relations = [
            word_pair_relationships[(idx, token_idx_r)] for idx in token_idxs_l
        ]
        word_pairs = [word_pair_dic[idx, token_idx_r] for idx in token_idxs_l]
        if match_score > 0:
            # SelectMostSimilarToken
            token_idx_l = np.argmax(sims)
        else:
            token_idx_l, score_sim, is_exists_same_word = select_most_related_token(
                sims, relations, word_pairs, th=0.86
            )
            # 採用すべき単語がなく、かつ、同じ単語がある場合は、ペア作成しない
            if token_idx_l is None and is_exists_same_word:
                continue
        pair_segment_list_cand.append(PairSegment(token_idx_l, token_idx_r, None))

    #  Calculate pseudo attribution scores as cosine similarity
    for pair_seg in pair_segment_list_cand:
        pair_seg.score = word_pair_sims.get((pair_seg.index_l, pair_seg.index_r), 0)

    # Greedy selection of top K pairs
    if match_score >= 0:
        pair_segment_list_cand = sorted(
            pair_segment_list_cand, key=lambda x: x.score, reverse=True
        )
    else:
        pair_segment_list_cand = sorted(pair_segment_list_cand, key=lambda x: x.score)
    pair_segment_list_filtered = []
    already_sel_l = set()
    already_sel_r = set()
    for pair_seg in pair_segment_list_cand:
        if len(pair_segment_list_filtered) >= topk:
            break
        if pair_seg.index_l is not None and pair_seg.index_l in already_sel_l:
            continue
        if pair_seg.index_r is not None and pair_seg.index_r in already_sel_r:
            continue
        if same_sign(match_score, pair_seg.score):
            already_sel_l.add(pair_seg.index_l)
            already_sel_r.add(pair_seg.index_r)
            pair_segment_list_filtered.append(pair_seg)

    return pair_segment_list_filtered


def extract_correlated_token_pairs_lime_pair(
    entity_pair: EntityPair,
    score_fn: Callable,
    topk: int,
    kernel: Callable,
    n_sample: int,
    random_state: int,
    fit_intercept: bool,
    batch_size: int = 512,
) -> List[PairSegment]:
    """対応したセグメントのリストを作成。score順に返す。"""
    pair_segment_list_cand: List[PairSegment] = []

    lime_result_org = LimeResult(
        *lime_make_explanation(
            entity_pair,
            score_fn,
            kernel=kernel,
            n_sample=n_sample,
            random_state=random_state,
            fit_intercept=fit_intercept,
        )
    )
    token_l_attr_score = {
        attr.index: attr.score for attr in lime_result_org.attributions_l
    }
    token_r_attr_score = {
        attr.index: attr.score for attr in lime_result_org.attributions_r
    }

    # Model’s match score for the record pair
    match_score = lime_result_org.match_score

    # Create all token pairs as pair_segment_list_cand
    for token_idx_l in range(entity_pair.entity_l.segment_size()):
        for token_idx_r in range(entity_pair.entity_r.segment_size()):
            pair_segment_list_cand.append(PairSegment(token_idx_l, token_idx_r, None))

    #  Calculate pseudo attribution scores
    for pair_seg in pair_segment_list_cand:
        score_l = (
            token_l_attr_score[pair_seg.index_l] if pair_seg.index_l is not None else 0
        )
        score_r = (
            token_r_attr_score[pair_seg.index_r] if pair_seg.index_r is not None else 0
        )
        pair_seg.score = score_l + score_r

    # Greedy selection of top K pairs
    if match_score >= 0:
        pair_segment_list_cand = sorted(
            pair_segment_list_cand, key=lambda x: x.score, reverse=True
        )
    else:
        pair_segment_list_cand = sorted(pair_segment_list_cand, key=lambda x: x.score)
    pair_segment_list_filtered = []
    already_sel_l = set()
    already_sel_r = set()
    for pair_seg in pair_segment_list_cand:
        if len(pair_segment_list_filtered) >= topk:
            break
        if pair_seg.index_l is not None and pair_seg.index_l in already_sel_l:
            continue
        if pair_seg.index_r is not None and pair_seg.index_r in already_sel_r:
            continue
        if same_sign(match_score, pair_seg.score):
            already_sel_l.add(pair_seg.index_l)
            already_sel_r.add(pair_seg.index_r)
            pair_segment_list_filtered.append(pair_seg)

    return pair_segment_list_filtered


def extract_correlated_token_pairs_lime_rank(
    entity_pair: EntityPair,
    score_fn: Callable,
    topk: int,
    kernel: Callable,
    n_sample: int,
    random_state: int,
    fit_intercept: bool,
    batch_size: int = 512,
) -> List[PairSegment]:
    """対応したセグメントのリストを作成。score順に返す。"""
    pair_segment_list_cand: List[PairSegment] = []

    lime_result_org = LimeResult(
        *lime_make_explanation(
            entity_pair,
            score_fn,
            kernel=kernel,
            n_sample=n_sample,
            random_state=random_state,
            fit_intercept=fit_intercept,
        )
    )
    token_l_attr_score = {
        attr.index: attr.score for attr in lime_result_org.attributions_l
    }
    token_r_attr_score = {
        attr.index: attr.score for attr in lime_result_org.attributions_r
    }

    # Model’s match score for the record pair
    match_score = lime_result_org.match_score

    # same ranked token でpair_segment_list_candを作成
    lime_result_org.attributions_l = sorted(
        lime_result_org.attributions_l, key=lambda x: abs(x.score), reverse=True
    )
    lime_result_org.attributions_r = sorted(
        lime_result_org.attributions_r, key=lambda x: abs(x.score), reverse=True
    )
    for rank in range(
        max(entity_pair.entity_l.segment_size(), entity_pair.entity_r.segment_size())
    ):
        token_idx_l = (
            lime_result_org.attributions_l[rank].index
            if rank < len(lime_result_org.attributions_l)
            else None
        )
        token_idx_r = (
            lime_result_org.attributions_r[rank].index
            if rank < len(lime_result_org.attributions_r)
            else None
        )
        pair_segment_list_cand.append(PairSegment(token_idx_l, token_idx_r, None))

    #  Calculate pseudo attribution scores
    for pair_seg in pair_segment_list_cand:
        score_l = (
            token_l_attr_score[pair_seg.index_l] if pair_seg.index_l is not None else 0
        )
        score_r = (
            token_r_attr_score[pair_seg.index_r] if pair_seg.index_r is not None else 0
        )
        pair_seg.score = score_l + score_r

    # Greedy selection of top K pairs
    if match_score >= 0:
        pair_segment_list_cand = sorted(
            pair_segment_list_cand, key=lambda x: x.score, reverse=True
        )
    else:
        pair_segment_list_cand = sorted(pair_segment_list_cand, key=lambda x: x.score)
    pair_segment_list_filtered = []
    already_sel_l = set()
    already_sel_r = set()
    for pair_seg in pair_segment_list_cand:
        if len(pair_segment_list_filtered) >= topk:
            break
        if pair_seg.index_l is not None and pair_seg.index_l in already_sel_l:
            continue
        if pair_seg.index_r is not None and pair_seg.index_r in already_sel_r:
            continue
        if same_sign(match_score, pair_seg.score):
            already_sel_l.add(pair_seg.index_l)
            already_sel_r.add(pair_seg.index_r)
            pair_segment_list_filtered.append(pair_seg)

    return pair_segment_list_filtered


def make_explanation(
    entity_pair: EntityPair,
    score_fn: Callable[[List[EntityPair]], np.ndarray],
    topk: int,
    *,
    kernel: Callable = kernel,
    n_sample: int = None,
    random_state: int = 0,
    fit_intercept: bool = True,
    batch_size: int = 512,
    method: str = "default",
) -> Tuple[LimeResultPair, EntityPair]:
    """Explain the prediction of the model using PINE."""
    # STEP1:Extracting the Top K Contributing Correlated Token Pairs
    if method == "default":
        step1_func = extract_correlated_token_pairs
    elif method == "cossim":
        step1_func = extract_correlated_token_pair_cossim
    elif method == "lime_pair":
        step1_func = extract_correlated_token_pairs_lime_pair
    elif method == "lime_rank":
        step1_func = extract_correlated_token_pairs_lime_rank
    else:
        raise ValueError(f"Invalid method: {method}")
    pair_segments = step1_func(
        entity_pair,
        score_fn,
        topk,
        kernel,
        n_sample,
        random_state,
        fit_intercept=fit_intercept,
        batch_size=batch_size,
    )
    merge_segments: List[MergedSegment] = []
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
        match_score = score_fn(entity_pair)[0]
        return LimeResultPair([], match_score, None, None, None), entity_pair_merged

    # STEP2: Calculating Attribution Scores via LIME
    lime_result_pair = LimeResultPair(
        *lime_make_explanation_without_separate_lr(
            entity_pair_merged, score_fn, fit_intercept=fit_intercept
        )
    )
    lime_result_pair.attributions = sorted(
        lime_result_pair.attributions, key=lambda x: abs(x.score), reverse=True
    )
    return lime_result_pair, entity_pair_merged
