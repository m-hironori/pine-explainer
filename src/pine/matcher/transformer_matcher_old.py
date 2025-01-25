import pathlib
import typing
from functools import partial

from transformers import AutoModelForSequenceClassification
from transformers.trainer_callback import TrainerState
from lemon.utils.matchers import TransformerMatcher

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


def _load_bert_mini_model_predict_func(
    dataset_name: str, model_root_dir: str = ".", batch_size: int = 512
) -> typing.Callable:
    # checkpoint のパスから、ベストモデルのパスを抽出
    checkpoints_path = (
        pathlib.Path(model_root_dir) / "bert-mini" / dataset_name / "checkpoints"
    )
    best_model_checkpoint_dir_path = _get_best_model_checkpoint_dir(checkpoints_path)
    if not best_model_checkpoint_dir_path.is_absolute():
        best_model_checkpoint_dir_path = (
            pathlib.Path(model_root_dir).parent / best_model_checkpoint_dir_path
        )
    # モデルをロード
    matcher = TransformerMatcher(
        str(best_model_checkpoint_dir_path),
        tokenizer_args={"model_max_length": 256},
        training_args={
            "per_device_train_batch_size": batch_size,
            "learning_rate": 3e-5,
            "warmup_steps": 50,
            "fp16": True,
            "num_train_epochs": 20,
        },
    )

    return partial(matcher.predict_proba, show_progress=False)


def make_transformer_matcher_func(
    dataset_name: str, model_root_dir: str, batch_size: int = 512
):
    predict_proba_func = _load_bert_mini_model_predict_func(
        dataset_name, model_root_dir, batch_size
    )
    proba_fn = _make_proba_fn(predict_proba_func)
    return proba_fn
