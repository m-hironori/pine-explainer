import sys
import pathlib
import typing
import math
import pickle
import concurrent.futures
import pandas as pd
import numpy as np

# Can not save and load the model created MalleganMatcher in lemon
# So, patching with these codes for saving and loading model file


# Magellan imports Tkinter, but we don't need it for our case
class DummyTkinterFrame:
    ...


class DummyTkinter:
    Frame = DummyTkinterFrame


sys.modules["Tkinter"] = DummyTkinter

import py_entitymatching.feature.autofeaturegen

del sys.modules["Tkinter"]


def _get_features_for_type_mod(column_type):
    """
    Get features to be generated for a type
    """
    # First get the look up table
    lookup_table = py_entitymatching.feature.autofeaturegen._get_feat_lkp_tbl()

    # Based on the column type, return the feature functions that should be
    # generated.
    if column_type == "str_eq_1w":
        features = lookup_table["STR_EQ_1W"]
    elif column_type == "str_bt_1w_5w":
        features = lookup_table["STR_BT_1W_5W"]
    elif column_type == "str_bt_5w_10w":
        features = lookup_table["STR_BT_5W_10W"]
    elif column_type == "str_gt_10w":
        features = lookup_table["STR_GT_10W"]
    elif column_type == "numeric":
        features = lookup_table["NUM"]
    elif column_type == "boolean":
        features = lookup_table["BOOL"]
    elif column_type == "un_determined":
        features = lookup_table["UN_DETERMINED"]
    else:
        raise TypeError("Unknown type")
    return features


# Monkey patching the function.
py_entitymatching.feature.autofeaturegen._get_features_for_type = (
    _get_features_for_type_mod
)

from lemon.utils.matchers import MagellanMatcher

from pine.matcher import _make_proba_fn


def _load_magellan_model_predict_func(
    dataset_name: str, model_root_dir: str = "."
) -> typing.Callable:
    with (
        pathlib.Path(model_root_dir) / "magellan" / dataset_name / "model.pickle"
    ).open("rb") as f:
        matcher: MagellanMatcher = pickle.load(f)
    return matcher.predict_proba


def make_magellan_matcher_func(dataset_name: str, model_root_dir: str):
    predict_proba_func = _load_magellan_model_predict_func(dataset_name, model_root_dir)
    proba_fn = _make_proba_fn(predict_proba_func)
    return proba_fn


def make_magellan_matcher_func_para(
    dataset_name: str, model_root_dir: str, para: int = 1
):
    # 並列処理数分のpredictorを作成
    funcs = [
        _load_magellan_model_predict_func(dataset_name, model_root_dir)
        for _ in range(para)
    ]

    def predict_proba_func(
        records_a: pd.DataFrame, records_b: pd.DataFrame, record_id_pairs: pd.DataFrame
    ):
        # para分にデータを分割
        each_len = math.ceil(len(record_id_pairs) / para)
        datas = []
        for i in range(0, len(record_id_pairs), each_len):
            datas.append(record_id_pairs.iloc[i : i + each_len])

        scores = [None] * len(datas)
        with concurrent.futures.ProcessPoolExecutor(max_workers=para) as executor:
            future_to_idx = {
                executor.submit(funcs[i], records_a, records_b, datas[i]): i
                for i in range(len(datas))
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    scores[idx] = pd.Series(
                        [np.nan] * len(datas[idx]), index=datas[idx].index
                    )
        scores = pd.concat(scores)
        return scores

    proba_fn = _make_proba_fn(predict_proba_func)
    return proba_fn
