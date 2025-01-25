from lemon.utils.datasets import SplittedDataset
import lemon.utils.datasets.deepmatcher
import pandas as pd


def _truncate_text(text, max_spaces=256):
    space_count = 0
    last_space_index = 0

    if text is None or pd.isna(text):
        return text

    for i, char in enumerate(text):
        if char == " ":
            space_count += 1
        if space_count >= max_spaces:
            last_space_index = i
            break

    return text[:last_space_index] if space_count >= max_spaces else text


def load_dataset(dataset_name: str, data_root_dir: str = ".") -> SplittedDataset:
    """lemonのutilを用いてデータセットをダウンロードしてメモリにロードする。data_root_dirにすでにデータがあればこれを使う。

    - データセットは以下のメンバ変数を持つ
        - (train|val|test).records.a = 左側データセットDataframe
        - (train|val|test).records.b = 右側データセットDataframe
        - (train|val|test).record_id_pairs = 比較するペアIDのDataframe
        - (train|val|test).labels = 比較結果のSeries
    - それぞれのデータは以下の形式
        - 左側データセットDataframe, 右側データセットDataframeは以下の形式
        - Dataframe.index : インデックス番号
        - Dataframe.columns : 各データセットの属性名リスト
    - 比較するペアDetaframeは以下の形式
        - Dataframe.index : ペアインデックス番号
        - Dataframe.columns : ["a.rid", "b.rid"]
            - "a.rid" : 左側データセットのインデックス番号
            - "b.rid" : 右側データセットのインデックス番号
    - 比較結果のSeries
        - Series.index : ペアインデックス番号
        - Series.values : True or False

    Args:
        dataset_name (str): データセット名
        data_root_dir (str, optional): データセットのルートディレクトリ名. Defaults to ".".

    Returns:
        lemon.utils.datasets.SplittedDataset: lemonのデータセットフォーマットのデータ
    """
    load_dataset_func = getattr(lemon.utils.datasets.deepmatcher, dataset_name)
    dataset = load_dataset_func(root=data_root_dir)
    # lemon.utils.datasets.deepmatcher.structured_dblp_acm has bug : Wrong attribute type setting. fix it.
    if dataset_name in [
        "structured_dblp_acm",
        "structured_dblp_google_scholar",
        "dirty_dblp_acm",
        "dirty_dblp_google_scholar",
    ]:
        for trainvaltest in ["train", "val", "test"]:
            dataset_sub = getattr(dataset, trainvaltest)
            dataset_sub.records.a["authors"] = dataset_sub.records.a["authors"].astype(
                "string"
            )
            dataset_sub.records.b["authors"] = dataset_sub.records.b["authors"].astype(
                "string"
            )
    # textual_company dataset, truncate each record to max 256 space-separated words
    if dataset_name == "textual_company":
        for trainvaltest in ["train", "val", "test"]:
            dataset_sub = getattr(dataset, trainvaltest)
            dataset_sub.records.a["content"] = dataset_sub.records.a["content"].apply(
                _truncate_text
            ).astype("string")
            dataset_sub.records.b["content"] = dataset_sub.records.b["content"].apply(
                _truncate_text
            ).astype("string")

    return dataset
