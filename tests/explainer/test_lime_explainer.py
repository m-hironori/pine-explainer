import numpy as np

from pine.entity import Attribute, Entity, EntityPair
from pine.explainer.lime_explainer import (
    kernel,
    make_data_with_max_mask_until_num_data,
    make_explanation,
    mask_segments,
)
from pine.matcher.magellan_matcher import make_magellan_matcher_func


def test_mask_segments():
    entity_l = Entity([Attribute("Name", "iphone 12", "string")])
    entity_r = Entity([Attribute("Name", "iphone 13 iphone", "string")])
    entity_pair = EntityPair(entity_l, entity_r)
    zs = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]])

    masked_entity_pair_list = mask_segments(entity_pair, zs)
    assert len(masked_entity_pair_list) == zs.shape[0]
    assert masked_entity_pair_list[0].is_equal_val(entity_pair)
    assert masked_entity_pair_list[1].is_equal_val(
        EntityPair(Entity([Attribute("Name", "12", "string")]), entity_r)
    )
    assert masked_entity_pair_list[2].is_equal_val(
        EntityPair(Entity([Attribute("Name", "iphone", "string")]), entity_r)
    )
    assert masked_entity_pair_list[3].is_equal_val(
        EntityPair(entity_l, Entity([Attribute("Name", "13", "string")]))
    )


def test_make_data_with_max_mask_until_num_data():
    dataset_name = "structured_amazon_google"
    model_root_dir = "data/lemon/model"
    predict_proba_func = make_magellan_matcher_func(dataset_name, model_root_dir)

    entity_l = Entity(
        [
            Attribute("title", "iphone 12", "string"),
            Attribute("manufacturer", "apple", "string"),
            Attribute("price", 100.0, "Float64"),
        ]
    )
    entity_r = Entity(
        [
            Attribute("title", "iphone 13 iphone", "string"),
            Attribute("manufacturer", "apple", "string"),
            Attribute("price", 100.0, "Float64"),
        ]
    )
    entity_pair = EntityPair(entity_l, entity_r)
    max_mask = 4
    num_data = 4

    zs, scores, distances = make_data_with_max_mask_until_num_data(
        entity_pair,
        max_mask,
        num_data,
        predict_proba_func,
        None,
        False,
        1,
    )

    assert zs.shape == (num_data, entity_pair.segment_size())
    assert scores.shape == (num_data, 1)
    assert distances.shape == (num_data,)
    return


def test_make_explanation():
    dataset_name = "structured_amazon_google"
    model_root_dir = "data/lemon/model"
    predict_proba_func = make_magellan_matcher_func(dataset_name, model_root_dir)

    entity_l = Entity(
        [
            Attribute("title", "iphone 12", "string"),
            Attribute("manufacturer", "apple", "string"),
            Attribute("price", 100.0, "Float64"),
        ]
    )
    entity_r = Entity(
        [
            Attribute("title", "iphone 13 iphone", "string"),
            Attribute("manufacturer", "apple", "string"),
            Attribute("price", 100.0, "Float64"),
        ]
    )
    entity_pair = EntityPair(entity_l, entity_r)

    (
        feature_scores_l,
        feature_scores_r,
        right,
        intercept,
        prediction_score,
        local_pred,
    ) = make_explanation(entity_pair, predict_proba_func, kernel, 1000, random_state=0)

    assert len(feature_scores_l) == entity_l.segment_size()
    assert len(feature_scores_r) == entity_r.segment_size()

    return
