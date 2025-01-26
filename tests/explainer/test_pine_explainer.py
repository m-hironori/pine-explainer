from pine.entity import Attribute, Entity, EntityPair
from pine.explainer.pine_explainer import (
    kernel,
    make_explanation,
)
from pine.matcher.magellan_matcher import make_magellan_matcher_func


def test_make_explanation():
    dataset_name = "structured_amazon_google"
    model_root_dir = "data/model"
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
    topk = 5

    lime_result_pair, entity_pair_merged = make_explanation(entity_pair, predict_proba_func, topk, n_sample=1000, random_state=0)

    assert len(lime_result_pair.attributions) <= topk
    assert len(lime_result_pair.attributions) == entity_pair_merged.segment_size()

    return
