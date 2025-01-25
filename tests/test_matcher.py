import pandas as pd
import numpy as np

from pine.entity import Attribute, Entity, EntityPair
from pine.matcher import _make_proba_fn
from pine.matcher.magellan_matcher import (
    _load_magellan_model_predict_func,
    make_magellan_matcher_func,
    make_magellan_matcher_func_para,
)
from pine.matcher.transformer_matcher import (
    load_transformer_matcher_func,
    make_transformer_matcher_func,
)

# from pine.matcher.transformer_matcher_old import (
#    _load_bert_mini_model_predict_func,
#    make_transformer_matcher_func,
# )


def test_load_bert_mini_model_predict_func():
    dataset_name = "structured_amazon_google"
    model_root_dir = "data/lemon/model"
    df_l = pd.DataFrame(
        columns=["title", "manufacturer", "price"],
        data=[["iphone 13 apple", "apple", 10], [None, None, None]],
        index=[0, 1],
    )
    df_r = pd.DataFrame(
        columns=["title", "manufacturer", "price"],
        data=[
            ["iphone 12", "apple", 10],
            ["iphone 13 apple", "apple", 10],
            [None, None, None],
        ],
        index=[0, 1, 2],
    )
    df_pair = pd.DataFrame(
        columns=["a.rid", "b.rid"],
        data=[[0, 0], [0, 1], [1, 0], [1, 2]],
        index=[0, 1, 2, 3],
    )
    df_pair.index.name = "pid"
    predict_proba_func = load_transformer_matcher_func(dataset_name, model_root_dir)
    ret = predict_proba_func(df_l, df_r, df_pair, 1)
    assert isinstance(ret, pd.Series)
    assert len(ret) == 4


def test_load_magellan_model_predict_func():
    dataset_name = "structured_amazon_google"
    model_root_dir = "data/lemon/model"
    df_l = pd.DataFrame(
        columns=["title", "manufacturer", "price"],
        data=[["iphone 13 apple", "apple", 10], [None, None, None]],
        index=[0, 1],
    )
    df_r = pd.DataFrame(
        columns=["title", "manufacturer", "price"],
        data=[
            ["iphone 12", "apple", 10],
            ["iphone 13 apple", "apple", 10],
            [None, None, None],
        ],
        index=[0, 1, 2],
    )
    df_pair = pd.DataFrame(
        columns=["a.rid", "b.rid"],
        data=[[0, 0], [0, 1], [1, 0], [1, 2]],
        index=[0, 1, 2, 3],
    )
    df_pair.index.name = "pid"
    predict_proba_func = _load_magellan_model_predict_func(dataset_name, model_root_dir)
    ret = predict_proba_func(
        df_l,
        df_r,
        df_pair,
    )

    assert isinstance(ret, pd.Series)
    assert len(ret) == 4


def test_make_proba_fn():
    dataset_name = "structured_amazon_google"
    model_root_dir = "data/lemon/model"
    predict_proba_func = _load_magellan_model_predict_func(dataset_name, model_root_dir)
    proba_fn = _make_proba_fn(predict_proba_func)

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

    scores = proba_fn([entity_pair, entity_pair])
    assert scores.shape == (2, 1)
    scores = proba_fn([entity_pair, entity_pair], False)
    assert scores.shape == (2,)


def test_make_magellan_matcher_func():
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
    dataset_name = "structured_amazon_google"
    model_root_dir = "data/lemon/model"
    matcher_fn = make_magellan_matcher_func(dataset_name, model_root_dir)

    scores = matcher_fn([entity_pair, entity_pair])
    assert scores.shape == (2, 1)
    scores = matcher_fn([entity_pair, entity_pair], False)
    assert scores.shape == (2,)


def test_make_magellan_matcher_func_para():
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
    dataset_name = "structured_amazon_google"
    model_root_dir = "data/lemon/model"
    matcher_fn = make_magellan_matcher_func_para(dataset_name, model_root_dir, 3)

    scores = matcher_fn([entity_pair, entity_pair])
    assert scores.shape == (2, 1)
    scores = matcher_fn([entity_pair, entity_pair], False)
    assert scores.shape == (2,)


def test_make_transformer_matcher_func():
    entity_l = Entity(
        [
            Attribute("title", "iphone 12", "string"),
            Attribute("manufacturer", "apple", "string"),
            Attribute("price", 100.0, "Float64"),
        ]
    )
    entity_r_1 = Entity(
        [
            Attribute("title", "iphone 13 iphone", "string"),
            Attribute("manufacturer", "apple", "string"),
            Attribute("price", 100.0, "Float64"),
        ]
    )
    entity_r_2 = Entity(
        [
            Attribute("title", "iphone", "string"),
            Attribute("manufacturer", "apple", "string"),
            Attribute("price", 100.0, "Float64"),
        ]
    )
    entity_pair_1 = EntityPair(entity_l, entity_r_1)
    entity_pair_2 = EntityPair(entity_l, entity_r_2)
    dataset_name = "structured_amazon_google"
    model_root_dir = "data/lemon/model"
    matcher_fn = make_transformer_matcher_func(dataset_name, model_root_dir)

    scores = matcher_fn([entity_pair_1, entity_pair_2])
    assert scores.shape == (2, 1)
    scores = matcher_fn([entity_pair_1, entity_pair_2], False)
    assert scores.shape == (2,)
    score_1 = matcher_fn([entity_pair_1], False)
    score_2 = matcher_fn([entity_pair_2], False)
    np.testing.assert_almost_equal(
        scores.astype(np.float16), np.array([score_1[0], score_2[0]], dtype=np.float16)
    )
