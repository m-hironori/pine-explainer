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


def load_transfofmer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_transformer_pred_func(model_name):
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
        ).to(device)
        return model(**encoded)

    return predict_func


def load_transformer_pred_func_trained(dataset_name, model_root_dir):
    checkpoints_path = (
        pathlib.Path(model_root_dir) / "bert-mini" / dataset_name / "checkpoints"
    )
    best_model_checkpoint_dir_path = _get_best_model_checkpoint_dir(checkpoints_path)
    if not best_model_checkpoint_dir_path.is_absolute():
        best_model_checkpoint_dir_path = (
            pathlib.Path(model_root_dir).parent / best_model_checkpoint_dir_path
        ).absolute()

    return load_transformer_pred_func(str(best_model_checkpoint_dir_path))


def _format_records(records, attr_strings):
    cols = list(records.columns)
    return pd.DataFrame(
        data={
            "record": [
                " ".join(
                    f"COL {attr_strings(i, c)} VAL {'' if pd.isna(v) else v}"
                    for c, v in zip(cols, r)
                )
                for i, r in enumerate(records.itertuples(index=False, name=None))
            ]
        },
        index=records.index,
        dtype="string",
    )


def load_transformer_matcher_func(
    dataset_name: str, model_root_dir: str, batch_size: int = None
):
    transformer_pred_func = load_transformer_pred_func_trained(
        dataset_name, model_root_dir
    )

    def predict_func(
        records_a: pd.DataFrame,
        records_b: pd.DataFrame,
        record_id_pairs: pd.DataFrame,
        batch_size: int = batch_size,
    ):
        record_pairs = (
            (
                record_id_pairs.merge(
                    records_a.add_prefix("a."), left_on="a.rid", right_index=True
                ).merge(records_b.add_prefix("b."), left_on="b.rid", right_index=True)
            )
            .sort_index()
            .drop(columns=["a.rid", "b.rid"])
        )
        attr_strings = [{}] * len(record_pairs)
        record_pairs = pd.concat(
            (
                _format_records(
                    record_pairs[
                        [c for c in record_pairs.columns if c.startswith("a.")]
                    ].rename(columns=lambda c: c[2:]),
                    lambda i, attr: attr_strings[i].get(("a", attr), attr),
                ).add_prefix("a."),
                _format_records(
                    record_pairs[
                        [c for c in record_pairs.columns if c.startswith("b.")]
                    ].rename(columns=lambda c: c[2:]),
                    lambda i, attr: attr_strings[i].get(("b", attr), attr),
                ).add_prefix("b."),
            ),
            axis=1,
        )

        outputs_batch_list = []
        batch_size = record_pairs.shape[0] if batch_size is None else batch_size
        for i in range(0, record_pairs.shape[0], batch_size):
            outputs_batch = transformer_pred_func(
                record_pairs.iloc[i : i + batch_size]["a.record"].tolist(),
                record_pairs.iloc[i : i + batch_size]["b.record"].tolist(),
            )["logits"].softmax(dim=1)
            outputs_batch = outputs_batch.detach().to("cpu").numpy()[:, 1]
            outputs_batch_list.append(outputs_batch)

        return pd.Series(
            np.concatenate(outputs_batch_list, axis=0), index=record_id_pairs.index
        )

    return predict_func


def make_transformer_matcher_func(
    dataset_name: str, model_root_dir: str, batch_size: int = 512
):
    predict_proba_func = load_transformer_matcher_func(dataset_name, model_root_dir, batch_size)
    proba_fn = _make_proba_fn(predict_proba_func)
    return proba_fn
