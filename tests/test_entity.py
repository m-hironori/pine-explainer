import pytest
import pandas as pd
from pine.entity import (
    regex_tokenizer,
    make_word_poslist,
    TokenPos,
    Attribute,
    Entity,
    EntityPair,
    SegmentPart,
    MergedSegment,
)


def test_regex_tokenizer():
    # tokenizer のテスト
    text = "COL  Name VAL iphone  12 "
    expected_tokens = ["COL", "Name", "VAL", "iphone", "12"]
    expected_token_poss = [
        TokenPos(start=0, end=3),
        TokenPos(start=5, end=9),
        TokenPos(start=10, end=13),
        TokenPos(start=14, end=20),
        TokenPos(start=22, end=24),
    ]
    tokens, token_poss = regex_tokenizer(text)
    assert tokens == expected_tokens
    assert token_poss == expected_token_poss


@pytest.mark.parametrize(
    "sentence, aggregate_same_word, stop_words, tokens_expected, token_poss_expected",
    [
        (
            "COL Name VAL iphone 12",
            True,
            ["COL", "VAL"],
            ["Name", "iphone", "12"],
            [
                [TokenPos(start=4, end=8)],
                [TokenPos(start=13, end=19)],
                [TokenPos(start=20, end=22)],
            ],
        ),
        (
            "COL Name VAL iphone 13 iphone",
            True,
            ["COL", "VAL"],
            ["Name", "iphone", "13"],
            [
                [TokenPos(start=4, end=8)],
                [TokenPos(start=13, end=19), TokenPos(start=23, end=29)],
                [TokenPos(start=20, end=22)],
            ],
        ),
        (
            "",
            True,
            None,
            [],
            [],
        ),
    ],
)
def test_make_word_poslist_parametrized(
    sentence, aggregate_same_word, stop_words, tokens_expected, token_poss_expected
):
    tokens, token_poss = make_word_poslist(
        sentence, regex_tokenizer, aggregate_same_word, stop_words
    )
    assert tokens == tokens_expected
    assert token_poss == token_poss_expected


def test_entity():
    # オブジェクト作成
    attr_list = [
        Attribute("Name", "iphone 13 iphone", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity = Entity(attr_list)

    # セグメントリスト
    expected_segment_list = [
        [SegmentPart(0, 0, 6), SegmentPart(0, 10, 16)],
        [SegmentPart(0, 7, 9)],
        [SegmentPart(1, None, None)],
    ]
    assert entity.segment_list == expected_segment_list

    # オブジェクト作成（値がない）
    attr_list = [
        Attribute("Name", "", "string"),
        Attribute("Price", None, "Float64"),
    ]
    entity = Entity(attr_list)

    # セグメントリスト（セグメントなし）
    expected_segment_list = []
    assert entity.segment_list == expected_segment_list


def test_entity_make_entity_by_deleting_segments():
    attr_list = [
        Attribute("Name", "iphone 13 iphone", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity = Entity(attr_list)

    # 文字セグメント削除
    entity_deleted = entity.make_entity_by_deleting_segments([0, 0])
    expected_attr_list = [
        Attribute("Name", "13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    expected_segment_list = [
        [SegmentPart(0, 0, 2)],
        [SegmentPart(1, None, None)],
    ]
    assert entity_deleted.attr_list == expected_attr_list
    assert entity_deleted.segment_list == expected_segment_list

    # 文字セグメントと文字以外セグメントを削除（文字列以外のセグメントは値はなくなるりセグメントもなくなる）
    entity_deleted = entity.make_entity_by_deleting_segments([2, 0])
    expected_attr_list = [
        Attribute("Name", "13", "string"),
        Attribute("Price", None, "Float64"),
    ]
    expected_segment_list = [
        [SegmentPart(0, 0, 2)],
    ]
    assert entity_deleted.attr_list == expected_attr_list
    assert entity_deleted.segment_list == expected_segment_list


def test_entity_to_dataframe():
    attr_list = [
        Attribute("Name", "iphone 13 iphone", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity = Entity(attr_list)
    # dataframeに変換
    df = entity.to_dataframe()
    expected_df = pd.DataFrame(
        data={"Name": ["iphone 13 iphone"], "Price": [10]}, columns=["Name", "Price"]
    )
    expected_df.index.name = "__id"
    expected_df["Name"] = expected_df["Name"].astype("string")
    expected_df["Price"] = expected_df["Price"].astype("Float64")
    pd.testing.assert_frame_equal(df, expected_df)


def test_from_dataframe():
    # dataframeから作成
    df = pd.DataFrame(
        data={"Name": ["iphone 13 iphone"], "Price": [10]}, columns=["Name", "Price"]
    )
    df["Name"] = df["Name"].astype("string")
    df["Price"] = df["Price"].astype("Float64")
    entity = Entity.from_dataframe(df)
    expected_attr_list = [
        Attribute("Name", "iphone 13 iphone", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    expected_segment_list = [
        [SegmentPart(0, 0, 6), SegmentPart(0, 10, 16)],
        [SegmentPart(0, 7, 9)],
        [SegmentPart(1, None, None)],
    ]
    assert entity.attr_list == expected_attr_list
    assert entity.segment_list == expected_segment_list

    # dataframeから作成
    df = pd.DataFrame(
        data={"Name": ["iphone 13 iphone"], "Price": [pd.NA]}, columns=["Name", "Price"]
    )
    df["Name"] = df["Name"].astype("string")
    df["Price"] = df["Price"].astype("Float64")
    entity = Entity.from_dataframe(df)
    expected_attr_list = [
        Attribute("Name", "iphone 13 iphone", "string"),
        Attribute("Price", None, "Float64"),
    ]
    expected_segment_list = [
        [SegmentPart(0, 0, 6), SegmentPart(0, 10, 16)],
        [SegmentPart(0, 7, 9)],
    ]
    assert entity.attr_list == expected_attr_list
    assert entity.segment_list == expected_segment_list


def test_get_attribute_list_by_segments():
    # attribute_listを取得
    attr_list = [
        Attribute("Name", "iphone 13 iphone", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity = Entity(attr_list)
    ret_attr_list = entity.get_attribute_list_by_segments([0])
    expected_attr_list = [
        Attribute("Name", "iphone iphone", "string"),
    ]
    assert ret_attr_list == expected_attr_list


def test_make_entity_by_adding_attribute():
    attr_list = [
        Attribute("Name", "13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity = Entity(attr_list)
    attr_list_add = [
        Attribute("Name", "iphone 13 iphone", "string"),
    ]
    entity_new = entity.make_entity_by_adding_attribute(attr_list_add)
    expected_entity = Entity(
        [
            Attribute("Name", "13 iphone 13 iphone", "string"),
            Attribute("Price", 10, "Float64"),
        ]
    )
    assert entity_new.attr_list == expected_entity.attr_list


def test_entity_pair():
    attr_list_l = [
        Attribute("Name", "iphone 13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l = Entity(attr_list_l)
    attr_list_r = [
        Attribute("Name", "iphone 12", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r = Entity(attr_list_r)
    entity_pair = EntityPair(entity_l, entity_r)
    expected_merged_segment_list = [
        MergedSegment([0], []),
        MergedSegment([1], []),
        MergedSegment([2], []),
        MergedSegment([], [0]),
        MergedSegment([], [1]),
        MergedSegment([], [2]),
    ]

    assert len(entity_pair.merged_segment_list) == len(expected_merged_segment_list)
    for merged_segment, expected_merged_segment in zip(
        entity_pair.merged_segment_list, expected_merged_segment_list
    ):
        assert (
            merged_segment.segment_list_in_l
            == expected_merged_segment.segment_list_in_l
        )
        assert (
            merged_segment.segment_list_in_r
            == expected_merged_segment.segment_list_in_r
        )

    assert entity_pair.segment_size() == len(expected_merged_segment_list)


def test_entity_pair_convert_segment_idx():
    attr_list_l = [
        Attribute("Name", "iphone 13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l = Entity(attr_list_l)
    attr_list_r = [
        Attribute("Name", "iphone 12", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r = Entity(attr_list_r)
    entity_pair = EntityPair(entity_l, entity_r)

    assert entity_pair.convert_segment_idx_from_l(1) == 1
    assert entity_pair.convert_segment_idx_from_l(2) == 2
    assert entity_pair.convert_segment_idx_from_r(1) == 4
    assert entity_pair.convert_segment_idx_from_r(2) == 5

    l_idxs, r_idxs = entity_pair.convert_segment_idx_to_entity_idx(1)
    assert l_idxs == [1] and r_idxs == []
    l_idxs, r_idxs = entity_pair.convert_segment_idx_to_entity_idx(3)
    assert l_idxs == [] and r_idxs == [0]


def test_entity_pair_get_segment_label():
    attr_list_l = [
        Attribute("Name", "iphone 13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l = Entity(attr_list_l)
    attr_list_r = [
        Attribute("Name", "iphone 12", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r = Entity(attr_list_r)
    entity_pair = EntityPair(entity_l, entity_r)

    assert entity_pair.get_segment_label(1) == ("13", "")
    assert entity_pair.get_segment_label(3) == ("", "iphone")

    (
        ret_entity_l,
        ret_entity_r,
        ret_label_l,
        ret_label_r,
    ) = entity_pair.get_segment_entity_label(1)
    assert ret_entity_l.is_equal_val(entity_l)
    assert ret_entity_r.is_equal_val(entity_r)
    assert ret_label_l == "13"
    assert ret_label_r == ""
    (
        ret_entity_l,
        ret_entity_r,
        ret_label_l,
        ret_label_r,
    ) = entity_pair.get_segment_entity_label(3)
    assert ret_entity_l.is_equal_val(entity_l)
    assert ret_entity_r.is_equal_val(entity_r)
    assert ret_label_l == ""
    assert ret_label_r == "iphone"


def test_entity_pair_is_equal_val():
    attr_list_l = [
        Attribute("Name", "iphone 13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l = Entity(attr_list_l)
    attr_list_r = [
        Attribute("Name", "iphone 12", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r = Entity(attr_list_r)
    entity_pair = EntityPair(entity_l, entity_r)

    attr_list_l_o = [
        Attribute("Name", "iphone 13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l_o = Entity(attr_list_l_o)
    attr_list_r_o = [
        Attribute("Name", "iphone 12", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r_o = Entity(attr_list_r_o)
    entity_pair_o = EntityPair(entity_l_o, entity_r_o)
    is_equal = entity_pair.is_equal_val(entity_pair_o)

    assert is_equal == True


def test_entity_pair_make_entity_pair_by_deleting_segments():
    attr_list_l = [
        Attribute("Name", "iphone 13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l = Entity(attr_list_l)
    attr_list_r = [
        Attribute("Name", "iphone 12", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r = Entity(attr_list_r)
    entity_pair = EntityPair(entity_l, entity_r)

    entity_pair_del = entity_pair.make_entity_pair_by_deleting_segments([0, 4])

    attr_list_l_o = [
        Attribute("Name", "13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l_o = Entity(attr_list_l_o)
    attr_list_r_o = [
        Attribute("Name", "iphone", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r_o = Entity(attr_list_r_o)
    entity_pair_o = EntityPair(entity_l_o, entity_r_o)

    assert entity_pair_del.is_equal_val(entity_pair_o)


def test_entity_pair_make_entity_pair_by_merging_segment_list():
    attr_list_l = [
        Attribute("Name", "iphone 13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l = Entity(attr_list_l)
    attr_list_r = [
        Attribute("Name", "iphone 12", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r = Entity(attr_list_r)
    entity_pair = EntityPair(entity_l, entity_r)
    merging_segment_list = [
        MergedSegment([2], [2]),
        MergedSegment([0], [0]),
    ]

    entity_pair_merged = entity_pair.make_entity_pair_by_merging_segment_list(
        merging_segment_list
    )

    expected_merged_segment_list = [
        MergedSegment([1], []),
        MergedSegment([], [1]),
        MergedSegment([0], [0]),
        MergedSegment([2], [2]),
    ]
    for merged_segment, expected_merged_segment in zip(
        entity_pair_merged.merged_segment_list, expected_merged_segment_list
    ):
        assert (
            merged_segment.segment_list_in_l
            == expected_merged_segment.segment_list_in_l
        )
        assert (
            merged_segment.segment_list_in_r
            == expected_merged_segment.segment_list_in_r
        )


def test_entity_pair_make_entity_pair_by_merging_segment_list_and_convert_segment_idx():
    attr_list_l = [
        Attribute("Name", "iphone 13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l = Entity(attr_list_l)
    attr_list_r = [
        Attribute("Name", "iphone 12", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r = Entity(attr_list_r)
    entity_pair = EntityPair(entity_l, entity_r)
    merging_segment_list = [
        MergedSegment([2], [2]),
        MergedSegment([0], [0]),
    ]
    entity_pair_merged = entity_pair.make_entity_pair_by_merging_segment_list(
        merging_segment_list
    )

    assert entity_pair_merged.convert_segment_idx_from_l(1) == 0
    assert entity_pair_merged.convert_segment_idx_from_l(2) == 3
    assert entity_pair_merged.convert_segment_idx_from_r(1) == 1
    assert entity_pair_merged.convert_segment_idx_from_r(2) == 3

    l_idxs, r_idxs = entity_pair_merged.convert_segment_idx_to_entity_idx(1)
    assert l_idxs == [] and r_idxs == [1]
    l_idxs, r_idxs = entity_pair_merged.convert_segment_idx_to_entity_idx(3)
    assert l_idxs == [2] and r_idxs == [2]


def test_entity_pair_make_entity_pair_by_merging_segment_list_and_get_segment_label():
    attr_list_l = [
        Attribute("Name", "iphone 13", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_l = Entity(attr_list_l)
    attr_list_r = [
        Attribute("Name", "iphone 12", "string"),
        Attribute("Price", 10, "Float64"),
    ]
    entity_r = Entity(attr_list_r)
    entity_pair = EntityPair(entity_l, entity_r)
    merging_segment_list = [
        MergedSegment([2], [2]),
        MergedSegment([0], [0]),
    ]
    entity_pair_merged = entity_pair.make_entity_pair_by_merging_segment_list(
        merging_segment_list
    )

    assert entity_pair_merged.get_segment_label(1) == ("", "12")
    assert entity_pair_merged.get_segment_label(3) == ("10", "10")

    (
        ret_entity_l,
        ret_entity_r,
        ret_label_l,
        ret_label_r,
    ) = entity_pair_merged.get_segment_entity_label(1)
    assert ret_entity_l.is_equal_val(entity_l)
    assert ret_entity_r.is_equal_val(entity_r)
    assert ret_label_l == ""
    assert ret_label_r == "12"
    (
        ret_entity_l,
        ret_entity_r,
        ret_label_l,
        ret_label_r,
    ) = entity_pair_merged.get_segment_entity_label(3)
    assert ret_entity_l.is_equal_val(entity_l)
    assert ret_entity_r.is_equal_val(entity_r)
    assert ret_label_l == "10"
    assert ret_label_r == "10"
