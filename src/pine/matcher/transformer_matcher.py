from typing import Callable, Tuple, List
import pathlib

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.trainer_callback import TrainerState
import pandas as pd
import numpy as np

from pine.matcher import _make_proba_fn
from pine.entity import Entity, EntityPair


def _get_best_model_checkpoint_dir(checkpoints_dir_path: pathlib.Path) -> pathlib.Path:
    # checkpoint の最後のものを探す
    last_checkpoints_dir = sorted(
        checkpoints_dir_path.iterdir(), key=lambda x: int(str(x).split("-")[-1])
    )[-1]
    # 訓練結果情報から、もっとの良いモデルのチェックポイントのパスを取得
    state = TrainerState.load_from_json(
        f"{str(last_checkpoints_dir)}/trainer_state.json"
    )
    return pathlib.Path((state.best_model_checkpoint))


def load_transfofmer(model_name)->Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_transformer_pred_func(model_name)->Callable[[List[str], List[str]], torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_transfofmer(model_name)
    model.to(device)
    model.eval()

    def predict_func(texta, textb):
        encoded = tokenizer(
            texta,
            textb,
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**encoded)
        return outputs

    return predict_func


def load_transformer_pred_func_trained(dataset_name, model_root_dir)->Callable[[List[str], List[str]], torch.Tensor]:
    checkpoints_path = (
        pathlib.Path(model_root_dir) / "bert-mini" / dataset_name / "checkpoints"
    )
    best_model_checkpoint_dir_path = _get_best_model_checkpoint_dir(checkpoints_path)
    if not best_model_checkpoint_dir_path.is_absolute():
        best_model_checkpoint_dir_path = (
            pathlib.Path(model_root_dir).parent / best_model_checkpoint_dir_path
        ).absolute()

    return load_transformer_pred_func(str(best_model_checkpoint_dir_path))


def entity_to_text(entity:Entity)->str:
    return " ".join(
        f"COL {attr.name} VAL {'' if pd.isna(attr.value) else attr.value}"
        for attr in entity.attr_list
    )

def make_transformer_matcher_func(
    dataset_name: str, model_root_dir: str, batch_size: int = 512
)->Callable[[List[EntityPair], bool], np.ndarray]:
    pred_func = load_transformer_pred_func_trained(dataset_name, model_root_dir)

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
                text_l = entity_to_text(entity_pair.entity_l)
                text_r = entity_to_text(entity_pair.entity_r)
                texts_l.append(text_l)
                texts_r.append(text_r)
            batch_preds = pred_func(texts_l, texts_r)["logits"].softmax(dim=1).to("cpu").numpy()[:, 1]
            all_probas.append(batch_preds)
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


def load_transformer_matcher_func(
    dataset_name: str, model_root_dir: str, batch_size: int = 512
)->Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.Series]:
    """lemonモジュール用のproba関数(0から1 の確率)を作成する。

    Args:
        dataset_name (str): データセット名
        model_root_dir (str): モデルのルートディレクトリ
        batch_size (int): バッチサイズ

    Returns:
        Callable[[pd.Dataframe, pd.Dataframe, pd.Dataframe], np.ndarray]: 本モジュール用のmatch score計算用関数(Dataframe 入力)

    """
    matcher_fnc_org = make_transformer_matcher_func(dataset_name, model_root_dir, batch_size)

    def proba_fn(
        records_a: pd.DataFrame,
        records_b: pd.DataFrame,
        record_id_pairs: pd.DataFrame,
        batch_size: int = batch_size,
    ):
        """Predict(0-1の範囲の確率)を出力する関数を作成
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


