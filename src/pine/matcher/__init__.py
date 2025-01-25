from typing import Callable, List

import numpy as np
import pandas as pd

from pine.entity import EntityPair


def _make_proba_fn(
    matcher_func: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]
) -> Callable[[List[EntityPair], bool], np.ndarray]:
    """lemonのmatcher.predict_proba関数を、本モジュール用の関数に合わせる

    Args:
        matcher_func (Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]): lemonのmatcherのpredict_proba関数

    Returns:
        Callable[[List[EntityPair], bool], np.ndarray]: 本モジュール用のmatch score計算用関数
    """    
    def proba_fn(
        entity_pairs: List[EntityPair], expand_axis: bool = True
    ) -> np.ndarray:
        dataframe_l_list = []
        dataframe_r_list = []
        for pair in entity_pairs:
            dataframe_l_list.append(pair.entity_l.to_dataframe())
            dataframe_r_list.append(pair.entity_r.to_dataframe())
        dataframe_l = pd.concat(dataframe_l_list).reset_index(drop=True)
        dataframe_r = pd.concat(dataframe_r_list).reset_index(drop=True)
        dataframe_id_pair = pd.DataFrame(
            {"a.rid": dataframe_l.index.tolist(), "b.rid": dataframe_r.index.tolist()}
        )
        dataframe_id_pair.index.name = "pid"
        df_score = matcher_func(dataframe_l, dataframe_r, dataframe_id_pair)
        scores = df_score.to_numpy()
        # スコアを規格化 0.0 - 1.0 を -1.0 - 1.0 にする
        scores = 2 * scores - 1.0
        if expand_axis:
            # limeでは、1データに複数のラベルの結果がある場合が想定されているため、一軸増増やしたデータを作成
            return scores[:, np.newaxis]
        return scores

    return proba_fn
