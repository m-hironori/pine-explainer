import pathlib
from typing import List, Callable

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)
import numpy as np
import pandas as pd

from pine.entity import Attribute, EntityPair, Entity


dataset_name_2_dir = {
    "structured_amazon_google": "deepmatcher/Structured/Amazon-Google",
    "structured_beer": "deepmatcher/Structured/Beer/exp_data",
    "structured_dblp_acm": "deepmatcher/Structured/DBLP-ACM/exp_data",
    "structured_dblp_google_scholar": "deepmatcher/Structured/DBLP-GoogleScholar/exp_data",
    "structured_fodors_zagat": "deepmatcher/Structured/Fodors-Zagats/exp_data",
    "structured_walmart_amazon": "deepmatcher/Structured/Walmart-Amazon/exp_data",
    "structured_itunes_amazon": "deepmatcher/Structured/iTunes-Amazon/exp_data",
    "dirty_dblp_acm": "deepmatcher/Dirty/DBLP-ACM/exp_data",
    "dirty_dblp_google_scholar": "deepmatcher/Dirty/DBLP-GoogleScholar/exp_data",
    "dirty_walmart_amazon": "deepmatcher/Dirty/Walmart-Amazon/exp_data",
    "dirty_itunes_amazon": "deepmatcher/Dirty/iTunes-Amazon/exp_data",
    "textual_abt_buy": "deepmatcher/Textual/Abt-Buy/exp_data",
    "textual_company": "deepmatcher/Textual/Company/exp_data",
    "wdc_products_small_cc50_un50": "wdc/Products_small_CC50_UN50",
    "wdc_products_medium_cc50_un50": "wdc/Products_medium_CC50_UN50",
}


class CLRTMatcherModel(torch.nn.Module):
    """A baseline model for EM."""

    def __init__(self, lm="roberta-base"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(lm)
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1):
        # x1 を BatchEncoding を想定
        outputs = self.bert(**x1)
        h1 = outputs[0][:, 0, :]
        pred = self.fc(h1)
        return pred


def _get_latest_model_file_path(model_dir_path: pathlib.Path) -> pathlib.Path:
    """model_dir_path 以下の全ての子供ディレクトリの中で matcher_model.pt を探して返す。"""
    model_file_paths = list(model_dir_path.glob("**/matcher_model.pt"))
    if not model_file_paths:
        raise FileNotFoundError(f"No matcher_model.pt found under {model_dir_path}")
    latest_model_file_path = sorted(
        model_file_paths, key=lambda x: x.stat().st_mtime, reverse=True
    )[0]
    return latest_model_file_path


def load_cler_matcher_model(dataset_name, model_root_dir):
    modek_dir_path = pathlib.Path(model_root_dir) / dataset_name_2_dir[dataset_name]
    model_path = _get_latest_model_file_path(modek_dir_path)
    print(f"model path {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLRTMatcherModel()
    model.to(device)

    # 学習済み重みのロード
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 推論用の設定(cuda なら fp16 にする)
    if device == "cuda":
        model = model.half()
        print("Using FP16 for Inference (CUDA)")
    else:
        print(f"Using FP32 for Inference ({device})")

    return model


def attr2text(record: List[Attribute]):
    """一つのrecordをテキストに変換
    Args:
        record (list): Attributeのリスト
    Returns:
        text (str): テキスト化したもの
    """
    texts = []
    for attr in record:
        if len(str(attr.value)) > 0 and (
            (attr.dtype == "string" or type(attr.value) == str)
            or (
                (attr.dtype == "float" or type(attr.value) == float)
                and not (attr.value is None or np.isnan(attr.value))
            )
        ):
            texts.append("COL %s VAL %s" % (attr.name, str(attr.value)))
    return " ".join(texts)


def make_cler_matcher_func(
    dataset_name: str, model_root_dir: str, batch_size: int = 512
) -> Callable[[List[EntityPair], bool], np.ndarray]:
    """pine.matcherモジュール用のscore関数(-1 から 1 の正規化済み)を作成する。

    Args:
        dataset_name (str): データセット名
        model_root_dir (str): モデルのルートディレクトリ
        batch_size (int): バッチサイズ

    Returns:
        Callable[[List[EntityPair], bool], np.ndarray]: 本モジュール用のmatch score計算用関数
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_cler_matcher_model(dataset_name, model_root_dir)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model.bert.config.name_or_path)

    def score_fn(
        entity_pairs: List[EntityPair], expand_axis: bool = True, is_proba: bool = False
    ) -> np.ndarray:
        """スコアを出力。エンティティペアごとに予測。
        Args:
            entity_pairs (list): 複数エンティティペア。
            expand_axis (bool): Trueなら"[[score],[score]]"の形で返す
            is_proba (bool): Trueなら0.0-1.0の確率で返す
        Returns:
            scores (np.ndarray): 各エンティティペアのスコア
        """
        all_probas = []
        for i in range(0, len(entity_pairs), batch_size):
            batch_pairs = entity_pairs[i : i + batch_size]

            # make text data
            texts_l, texts_r = [], []
            for entity_pair in batch_pairs:
                record_l = entity_pair.entity_l
                record_r = entity_pair.entity_r
                text_l = attr2text(record_l.attr_list)
                text_r = attr2text(record_r.attr_list)
                texts_l.append(text_l)
                texts_r.append(text_r)
            # predict proba
            encoded = tokenizer(
                text=texts_l,
                text_pair=texts_r,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                encoded.to(device)
                logits = model(encoded)
                batch_probas = torch.softmax(logits, dim=1)
                batch_probas = (
                    batch_probas.detach().cpu().numpy()[:, 1]
                )  # positiveクラスのスコアを取得
                all_probas.append(batch_probas)

        probas = np.concatenate(all_probas, axis=0)

        if is_proba:
            scores = probas
        else:
            # スコアを規格化 0.0 - 1.0 を -1.0 - 1.0 にする
            scores = 2 * probas - 1.0

        # limeでは、1データに複数のラベルの結果がある場合が想定されているため、一軸増増やしたデータを作成
        if expand_axis:
            return scores[:, np.newaxis]
        return scores

    return score_fn


def make_cler_matcher_proba_input_df(
    dataset_name: str, model_root_dir: str, batch_size: int = 512
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.Series]:
    """lemonモジュール用のproba関数(0から1 の確率)を作成する。

    Args:
        dataset_name (str): データセット名
        model_root_dir (str): モデルのルートディレクトリ
        batch_size (int): バッチサイズ

    Returns:
        Callable[[pd.Dataframe, pd.Dataframe, pd.Dataframe], np.ndarray]: 本モジュール用のmatch score計算用関数(Dataframe 入力)

    """
    matcher_fnc_org = make_cler_matcher_func(dataset_name, model_root_dir, batch_size)

    def proba_fn(
        records_a: pd.DataFrame,
        records_b: pd.DataFrame,
        record_id_pairs: pd.DataFrame,
        batch_size: int = batch_size,
    ) -> pd.Series:
        """スコアを出力。
        Args:
            records_a (pd.DataFrame): エンティティAのデータフレーム
            records_b (pd.DataFrame): エンティティBのデータフレーム
            record_id_pairs (pd.DataFrame): レコードIDペアのデータフレーム
            batch_size (int): バッチサイズ
        Returns:
            scores (pd.Series): レコードIDペアのスコア
        """
        all_probas = []
        for i in range(0, len(record_id_pairs), batch_size):
            batch_pairs = record_id_pairs[i : i + batch_size]
            entity_pairs = []
            for _, row in batch_pairs.iterrows():
                record_a = records_a.loc[[row["a.rid"]]]
                record_b = records_b.loc[[row["b.rid"]]]
                entity_pair = EntityPair(
                    Entity.from_dataframe(record_a), Entity.from_dataframe(record_b)
                )
                entity_pairs.append(entity_pair)
            batch_probas = matcher_fnc_org(
                entity_pairs, expand_axis=False, is_proba=True
            )
            all_probas.append(batch_probas)

        probas = np.concatenate(all_probas, axis=0)
        return pd.Series(probas, index=record_id_pairs.index)

    return proba_fn
