from pathlib import Path

from lemon.utils.datasets import SplittedDataset
import lemon.utils.datasets.deepmatcher
import pandas as pd


def load_wdc_dataset_for_lemon_format(
    size_name: str,
    corner_case: int,
    unseen: int,
    dataset_root_dir: str,
) -> SplittedDataset:
    """Load WDC products dataset in LEMon format.
    Args:
        size_name (str): Size name of the WDC products dataset (e.g., "small", "medium", "large").
        corner_case (int): Corner case identifier.
        unseen (int): Unseen identifier.
        dataset_root_dir (str): Root directory where the dataset is stored.
    Returns:
        SplittedDataset: The loaded WDC products dataset in LEMon format.
    """
    assert size_name in [
        "small",
        "medium",
        "large",
    ], "size_name must be 'small', 'medium', or 'large'"
    assert corner_case in [80, 50, 20], "corner_case must be 80, 50 or 20"
    assert unseen in [0, 50, 100], "corner_case must be 0, 50 or 100"

    data_dir_path = (
        Path(dataset_root_dir)
        / "wdc"
        / f"Products_{size_name}_CC{corner_case}_UN{unseen}"
    )

    dtypes = {
        "id": "int64",
        "brand": "string",
        "title": "string",
        "description": "string",
        "price": "string",
        "priceCurrency": "string",
    }
    records_a, records_b = [
        pd.read_csv(data_dir_path / f, index_col="id", dtype=dtype).rename_axis(
            index="__id"
        )
        for f, dtype in [("tableA.csv", dtypes), ("tableB.csv", dtypes)]
    ]
    train_pairs, val_pairs, test_pairs = [
        pd.read_csv(data_dir_path / f)
        .rename(columns={"ltable_id": "a.rid", "rtable_id": "b.rid"})
        .astype({"label": "bool"})
        .rename_axis(index="pid")
        for f in ["train.csv", "valid.csv", "test.csv"]
    ]
    return SplittedDataset(
        records=(records_a, records_b),
        record_id_pairs_train=train_pairs[["a.rid", "b.rid"]],
        record_id_pairs_val=val_pairs[["a.rid", "b.rid"]],
        record_id_pairs_test=test_pairs[["a.rid", "b.rid"]],
        labels_train=train_pairs["label"],
        labels_val=val_pairs["label"],
        labels_test=test_pairs["label"],
    )


def wdc_products_small_cc50_un50(root: str = None):
    return load_wdc_dataset_for_lemon_format(
        size_name="small",
        corner_case=50,
        unseen=50,
        dataset_root_dir=root if root is not None else "data",
    )


def wdc_products_medium_cc50_un50(root: str = None):
    return load_wdc_dataset_for_lemon_format(
        size_name="medium",
        corner_case=50,
        unseen=50,
        dataset_root_dir=root if root is not None else "data",
    )


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
    # WDC データセット
    if dataset_name == "wdc_products_small_cc50_un50":
        return wdc_products_small_cc50_un50(root=data_root_dir)
    elif dataset_name == "wdc_products_medium_cc50_un50":
        return wdc_products_medium_cc50_un50(root=data_root_dir)

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
            dataset_sub.records.a["content"] = (
                dataset_sub.records.a["content"].apply(_truncate_text).astype("string")
            )
            dataset_sub.records.b["content"] = (
                dataset_sub.records.b["content"].apply(_truncate_text).astype("string")
            )

    return dataset
