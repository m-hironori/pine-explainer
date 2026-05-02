import contextlib
import logging
import sys
import pathlib
import typing
import math
import pickle
import concurrent.futures
import pandas as pd
import numpy as np


@contextlib.contextmanager
def _suppress_cloudpickle_module_scan_warnings():
    # py_entitymatching generates feature functions via exec() with __module__=None.
    # cloudpickle (used by LIME) scans all sys.modules to locate these functions,
    # hitting transformers 5.x's _fast image-processor aliases and triggering
    # spurious "alias will be removed" warnings. Filter only those messages.
    class _Filter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return not ("image_processing" in msg and "alias will be removed" in msg)

    logger = logging.getLogger("transformers")
    f = _Filter()
    logger.addFilter(f)
    try:
        yield
    finally:
        logger.removeFilter(f)

# Can not save and load the model created MalleganMatcher.
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


def load_magellan_model_predict_func(
    dataset_name: str, model_root_dir: str = "."
) -> typing.Callable:
    with (
        pathlib.Path(model_root_dir) / "magellan" / dataset_name / "model.pickle"
    ).open("rb") as f:
        matcher: MagellanMatcher = pickle.load(f)
    return matcher.predict_proba


def make_magellan_matcher_func(dataset_name: str, model_root_dir: str):
    predict_proba_func = load_magellan_model_predict_func(dataset_name, model_root_dir)
    base_proba_fn = _make_proba_fn(predict_proba_func)

    def proba_fn(entity_pairs, expand_axis=True):
        with _suppress_cloudpickle_module_scan_warnings():
            return base_proba_fn(entity_pairs, expand_axis)

    return proba_fn


