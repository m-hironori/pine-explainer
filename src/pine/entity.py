from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class TokenPos:
    """文字位置データ"""

    start: int
    end: int


def regex_tokenizer(text: str, sep_regex: str = "\s+") -> Tuple[int, int]:
    """文字列を入力に、regex区切り(デフォルトは空白)の開始、終了文字位置IDXを返す"""
    # 各tokenの表層文字列
    tokens = []
    # 各tokenの文字IDXペア(開始,終了)リスト
    token_poss = []
    pattern = re.compile(sep_regex)
    pos = 0
    while pos < len(text):
        m = pattern.search(text, pos)
        if m is None:
            break
        token_poss.append(TokenPos(pos, m.start()))
        tokens.append(text[pos : m.start()])
        pos = m.end()
    if pos < len(text):
        token_poss.append(TokenPos(pos, len(text)))
        tokens.append(text[pos : len(text)])
    return tokens, token_poss


def make_word_poslist(
    item: str,
    tokenizer: Callable,
    aggregate_same_word: bool = True,
    stop_words: List[str] = None,
) -> Tuple[List[str], List[List[TokenPos]]]:
    """文字列を入力に、単語リストとその位置リストを返す。
    単語リストのインデックスと位置リストのインデックスは、その単語の位置をさす。
    位置リストの各要素は、位置のリストであり、aggregate_same_word = True の場合、同じ単語が複数出現した場合、複数の位置が入る。

    Arguments:
    * item : item文字列
    * tokenizer : 文字列を入力に、単語列と各単語の開始文字IDXと終了文字IDXを出力する関数。
    * aggregate_same_word(bool) : 同じ単語を一つにまとめる
    * stop_word (List[str]) : 必要ない単語リスト

    Return:
    * words = itemの単語リスト
    * pos_list = wordsの各要素の位置リスト
    """
    tokens, token_poss = tokenizer(item)
    # 必要ない単語を除く
    if stop_words:
        idx_filtered = list(
            filter(lambda x: tokens[x] not in stop_words, range(len(tokens)))
        )
        tokens = [tokens[i] for i in idx_filtered]
        token_poss = [token_poss[i] for i in idx_filtered]
    # idx -> [単語開始IDX、単語終了IDX]のリスト
    token_poss = [[token_pos] for token_pos in token_poss]
    # 同じ表層文字列の単語をマージする
    if aggregate_same_word:
        tokens_merged: List[str] = []
        token_poss_merged: List[List[TokenPos]] = []
        for idx, (token, token_poss) in enumerate(zip(tokens, token_poss)):
            if token in tokens_merged:
                idx = tokens_merged.index(token)
                token_poss_merged[idx].extend(token_poss)
            else:
                tokens_merged.append(token)
                token_poss_merged.append(token_poss)
        tokens = tokens_merged
        token_poss = token_poss_merged

    return tokens, token_poss


@dataclass
class Attribute:
    """Entityの属性データ"""

    name: str
    value: Any
    dtype: str


@dataclass
class SegmentPart:
    """セグメントのパーツデータ"""

    attr_index: int
    start: Optional[int]
    end: Optional[int]


class Entity:
    """Entityを表すクラス"""

    def __init__(self, attr_list: List[Attribute] = []):
        self.attr_list: List[Attribute] = attr_list
        self.segment_list: List[List[SegmentPart]] = self._make_segments(attr_list)

    def _make_segments(self, attr_list: List[Attribute]) -> List[List[SegmentPart]]:
        """セグメントを作成する"""
        segment_list: List[List[SegmentPart]] = []
        word_to_seg: Dict[str, List[SegmentPart]] = {}
        for attr_index, attr in enumerate(attr_list):
            if attr.dtype == "string":
                # 文字列の場合は、単語区切り
                words, poslist = make_word_poslist(attr.value, regex_tokenizer, True)
                for word, poss in zip(words, poslist):
                    segment = [
                        SegmentPart(attr_index, pos.start, pos.end) for pos in poss
                    ]
                    if word in word_to_seg:
                        # すでに単語がある場合、セグメントにセグメントパーツを追加
                        word_to_seg[word].extend(segment)
                    else:
                        # 初めて出てくる単語の場合、セグメントを新規追加
                        word_to_seg[word] = segment
                        segment_list.append(segment)
            else:
                # 文字列の以外の場合は、値全体
                if attr.value is None:
                    # 値がない場合はセグメントを作成しない
                    continue
                segment = [SegmentPart(attr_index, None, None)]
                segment_list.append(segment)
        return segment_list

    def segment_size(self) -> int:
        """セグメントのサイズを返す"""
        return len(self.segment_list)

    def get_segment_label(self, index: int) -> str:
        """index番号のセグメントの値(ラベル)を返す"""
        segment_parts = self.segment_list[index]
        # 一つのセグメントのラベルは共通なので、一番はじめのセグメントパーツのラベルを返す
        attr = self.attr_list[segment_parts[0].attr_index]
        # 文字列の場合、その位置の単語を返す
        if attr.dtype == "string":
            return attr.value[segment_parts[0].start : segment_parts[0].end]
        # 文字列ではない場合、NoneならNoneを、それ以外はstrにして返す
        if attr.value is None:
            return None
        return str(attr.value)

    def get_attribute_list_by_segments(self, index_list: List[int]) -> List[Attribute]:
        """index番号リストのセグメントのAttributeを返す。
        Attributeのtypeがstringの場合、valueは元データの出現順を保つ。"""
        # 対象部分を attr_index 毎にまとめる
        value_seg_parts: Dict[int, List[SegmentPart]] = {}
        for seg_idx in index_list:
            for seg_part in self.segment_list[seg_idx]:
                if seg_part.attr_index not in value_seg_parts:
                    value_seg_parts[seg_part.attr_index] = []
                value_seg_parts[seg_part.attr_index].append(seg_part)
        ret_attr_list = []

        for target_attr_idx, target_segparts in value_seg_parts.items():
            attr_org = self.attr_list[target_attr_idx]
            # 文字列ではない場合、値を抽出
            if attr_org.dtype != "string":
                attr = Attribute(attr_org.name, attr_org.value, attr_org.dtype)
                ret_attr_list.append(attr)
                continue
            val = ""
            target_segparts = sorted(target_segparts, key=lambda x: x.start)
            for target_segpart in target_segparts:
                val += " " + attr_org.value[target_segpart.start : target_segpart.end]
            # 連続空白は一つの空白に変更
            val = re.sub("\s+", " ", val)
            # 前後の空白は削除
            val = val.strip(" ")
            attr = Attribute(attr_org.name, val, attr_org.dtype)
            ret_attr_list.append(attr)
        return ret_attr_list

    def make_entity_by_deleting_segments(
        self, index_list: List[int], mask_token_str: str = None
    ) -> Entity:
        """インデックス番号のセグメントを消したEntityを返す"""
        if mask_token_str is None:
            mask_token_str = ""

        # 同じインデックスを削除
        index_list = sorted(set(index_list))

        attr_list = copy.deepcopy(self.attr_list)
        # 削除対象部分を attr_index 毎にまとめる
        del_value_seg_parts: Dict[int, List[SegmentPart]] = {}
        for seg_idx in index_list:
            for seg_part in self.segment_list[seg_idx]:
                if seg_part.attr_index not in del_value_seg_parts:
                    del_value_seg_parts[seg_part.attr_index] = []
                del_value_seg_parts[seg_part.attr_index].append(seg_part)
        # attr_index 毎にはじめから順番に削除する
        for target_attr_idx, target_segparts in del_value_seg_parts.items():
            # 文字列ではない場合、値をNoneにして終了
            if target_segparts[0].start is None or target_segparts[0].end is None:
                attr_list[target_attr_idx].value = None
                continue
            # 文字列の場合、はじめから順番に消していく
            deleted_len = 0
            val = attr_list[target_attr_idx].value
            target_segparts = sorted(target_segparts, key=lambda x: x.start)
            for target_segpart in target_segparts:
                val = (
                    val[: target_segpart.start - deleted_len]
                    + mask_token_str
                    + val[target_segpart.end - deleted_len :]
                )
                deleted_len += (
                    target_segpart.end - target_segpart.start - len(mask_token_str)
                )
            # 連続空白は一つの空白に変更
            val = re.sub("\s+", " ", val)
            # 前後の空白は削除
            val = val.strip(" ")
            attr_list[target_attr_idx].value = val
        return Entity(attr_list)

    def make_entity_by_adding_attribute(self, attr_list: List[Attribute]) -> Entity:
        """"""
        org_attr_list = copy.deepcopy(self.attr_list)
        for attr in attr_list:
            is_find = False
            for org_attr in org_attr_list:
                # Attribute name と dtype が同じか確認
                if org_attr.name == attr.name and org_attr.dtype == attr.dtype:
                    is_find = True
                    if org_attr.dtype == "string":
                        org_attr.value += " " + attr.value
                        # 前後の空白は削除
                        org_attr.value = org_attr.value.strip(" ")
                    else:
                        org_attr.value = attr.value
            if is_find == False:
                ValueError(
                    "Can not add the attribute "
                    f"name={attr.name} val={attr.value} dtype={attr.dtype}"
                )
        return Entity(org_attr_list)

    def is_equal_val(self, entity_other: Entity) -> bool:
        """値が同じか"""
        # attr_list が同じか
        if len(self.attr_list) != len(entity_other.attr_list):
            return False
        for attr, attr_o in zip(self.attr_list, entity_other.attr_list):
            if attr != attr_o:
                return False
        # segment_list が同じか
        if len(self.segment_list) != len(entity_other.segment_list):
            return False
        for seg, seg_o in zip(self.segment_list, entity_other.segment_list):
            if seg != seg_o:
                return False
        return True

    def to_dataframe(self) -> pd.DataFrame:
        """EntityをDataFrame表現に変換する"""
        df = pd.DataFrame(
            data={attr.name: [attr.value] for attr in self.attr_list},
            columns=[attr.name for attr in self.attr_list],
        )
        df.index.name = "__id"
        for col, attr in zip(df.columns, self.attr_list):
            df[col] = df[col].astype(attr.dtype)
        return df

    def from_dataframe(df: pd.DataFrame) -> Entity:
        """DataFrameから作成"""
        if len(df) != 1:
            ValueError("DataFrame must have only 1 record.")
        attr_list: List[Attribute] = []
        for col, dtype in df.dtypes.items():
            val = df[col].iat[0]
            if pd.isna(val):
                val = None
                if dtype == "string":
                    val = ""
            attr = Attribute(col, val, dtype)
            attr_list.append(attr)
        return Entity(attr_list)


@dataclass
class MergedSegment:
    segment_list_in_l: List[int] = field(default_factory=lambda: [])
    segment_list_in_r: List[int] = field(default_factory=lambda: [])


class EntityPair:
    """EntityPairクラス"""

    def __init__(self, entity_l: Entity, entity_r: Entity) -> None:
        self.entity_l = entity_l
        self.entity_r = entity_r
        self.merged_segment_list: List[MergedSegment] = []
        # 初期セグメントは左と右の順番
        for idx in range(self.entity_l.segment_size()):
            self.merged_segment_list.append(MergedSegment([idx], []))
        for idx in range(self.entity_r.segment_size()):
            self.merged_segment_list.append(MergedSegment([], [idx]))

    def segment_size(self):
        """セグメントサイズを返す"""
        return len(self.merged_segment_list)

    def convert_segment_idx_from_l(self, l_index: int) -> int:
        """左エンティティのインデックスをもとに、EntityPairのセグメントインデックスを返す"""
        idx = None
        for i, merged_segment in enumerate(self.merged_segment_list):
            if l_index in merged_segment.segment_list_in_l:
                idx = i
                break
        return idx

    def convert_segment_idx_from_r(self, r_index: int) -> int:
        """右エンティティのインデックスをもとに、EntityPairのセグメントインデックスを返す"""
        idx = None
        for i, merged_segment in enumerate(self.merged_segment_list):
            if r_index in merged_segment.segment_list_in_r:
                idx = i
                break
        return idx

    def convert_segment_idx_to_entity_idx(
        self, index: int
    ) -> Tuple[List[int], List[int]]:
        """セグメントインデックスをもとに、右エンティティのセグメントリスト、左エンティティのインデックスリストを返す"""
        merged_segment = self.merged_segment_list[index]
        return merged_segment.segment_list_in_l, merged_segment.segment_list_in_r

    def get_segment_label(self, index: int) -> Tuple[str, str]:
        """セグメントインデックスのラベルを返す"""
        labels_l, labels_r = [], []
        merged_segment = self.merged_segment_list[index]
        for l_idx in sorted(merged_segment.segment_list_in_l):
            labels_l.append(self.entity_l.get_segment_label(l_idx))
        for r_idx in sorted(merged_segment.segment_list_in_r):
            labels_r.append(self.entity_r.get_segment_label(r_idx))
        return ",".join(labels_l), ",".join(labels_r)

    def get_segment_entity_label(self, index: int) -> Tuple[Entity, Entity, str, str]:
        """セグメントインデックスのインティティとラベルを返す"""
        labels_l, labels_r = [], []
        merged_segment = self.merged_segment_list[index]
        for l_idx in sorted(merged_segment.segment_list_in_l):
            labels_l.append(self.entity_l.get_segment_label(l_idx))
        for r_idx in sorted(merged_segment.segment_list_in_r):
            labels_r.append(self.entity_r.get_segment_label(r_idx))
        return self.entity_l, self.entity_r, ",".join(labels_l), ",".join(labels_r)

    def is_equal_val(self, other: EntityPair) -> bool:
        """値が同じか"""
        if not self.entity_l.is_equal_val(other.entity_l):
            return False
        if not self.entity_r.is_equal_val(other.entity_r):
            return False
        if len(self.merged_segment_list) != len(other.merged_segment_list):
            return False
        for idx in range(len(self.merged_segment_list)):
            if (
                self.merged_segment_list[idx].segment_list_in_l
                != other.merged_segment_list[idx].segment_list_in_l
                or self.merged_segment_list[idx].segment_list_in_r
                != other.merged_segment_list[idx].segment_list_in_r
            ):
                return False
        return True

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """データフレームを作成する。左と右それぞれ作成"""
        return self.entity_l.to_dataframe(), self.entity_r.to_dataframe()

    def make_entity_pair_by_deleting_segments(
        self, index_list: List[int], mask_token_str: str = None
    ) -> EntityPair:
        """インデックス番号のセグメントを消したEntityPairを返す"""
        # 削除対象を抽出
        index_list_l: List[int] = []
        index_list_r: List[int] = []
        for idx in index_list:
            index_list_l.extend(self.merged_segment_list[idx].segment_list_in_l)
            index_list_r.extend(self.merged_segment_list[idx].segment_list_in_r)
        index_list_l = sorted(set(index_list_l))
        index_list_r = sorted(set(index_list_r))
        # エンティティから削除
        entity_l_deleted = self.entity_l.make_entity_by_deleting_segments(
            index_list_l, mask_token_str
        )
        entity_r_deleted = self.entity_r.make_entity_by_deleting_segments(
            index_list_r, mask_token_str
        )
        entity_pair_new = EntityPair(entity_l_deleted, entity_r_deleted)
        # 結合セグメントから削除しながら新しい結合セグメントを作成
        index_map_l = {}
        del_num = 0
        for org_idx in range(self.entity_l.segment_size()):
            if org_idx in index_list_l:
                index_map_l[org_idx] = None
                del_num += 1
            else:
                index_map_l[org_idx] = org_idx - del_num
        index_map_r = {}
        del_num = 0
        for org_idx in range(self.entity_r.segment_size()):
            if org_idx in index_list_r:
                index_map_r[org_idx] = None
                del_num += 1
            else:
                index_map_r[org_idx] = org_idx - del_num
        merged_segment_list_new = []
        for merged_segment in self.merged_segment_list:
            merged_segment_new = MergedSegment()
            for l_idx in merged_segment.segment_list_in_l:
                if index_map_l[l_idx] is not None:
                    merged_segment_new.segment_list_in_l.append(index_map_l[l_idx])
            for r_idx in merged_segment.segment_list_in_r:
                if index_map_r[r_idx] is not None:
                    merged_segment_new.segment_list_in_r.append(index_map_r[r_idx])
            if (
                len(merged_segment_new.segment_list_in_l) != 0
                or len(merged_segment_new.segment_list_in_r) != 0
            ):
                merged_segment_list_new.append(merged_segment_new)
        entity_pair_new.merged_segment_list = merged_segment_list_new
        entity_pair_new.merged_segment_list = (
            entity_pair_new._sort_merged_segment_list()
        )
        return entity_pair_new

    def make_entity_pair_by_merging_segment_list(
        self, merging_segment_list: List[MergedSegment]
    ) -> EntityPair:
        """"""
        entity_l_new = copy.deepcopy(self.entity_l)
        entity_r_new = copy.deepcopy(self.entity_r)
        entity_pair_new = EntityPair(entity_l_new, entity_r_new)
        # 既存のセグメントからインデックスを削除する
        for merged_segment in merging_segment_list:
            for l_idx in merged_segment.segment_list_in_l:
                for merged_segment_new in entity_pair_new.merged_segment_list:
                    if l_idx in merged_segment_new.segment_list_in_l:
                        merged_segment_new.segment_list_in_l.remove(l_idx)
            for r_idx in merged_segment.segment_list_in_r:
                for merged_segment_new in entity_pair_new.merged_segment_list:
                    if r_idx in merged_segment_new.segment_list_in_r:
                        merged_segment_new.segment_list_in_r.remove(r_idx)
        # 空になったmerged_segmentを削除する
        empty_segment_list = []
        for merged_segment_new in entity_pair_new.merged_segment_list:
            if (
                len(merged_segment_new.segment_list_in_l) == 0
                and len(merged_segment_new.segment_list_in_r) == 0
            ):
                empty_segment_list.append(merged_segment_new)
        for empty_segment in empty_segment_list:
            entity_pair_new.merged_segment_list.remove(empty_segment)
        # 最後にmerged_segmentを追加する
        entity_pair_new.merged_segment_list.extend(merging_segment_list)
        # 並べ替える
        entity_pair_new.merged_segment_list = (
            entity_pair_new._sort_merged_segment_list()
        )
        return entity_pair_new


    def make_entity_pair_by_merging_segment_list_only(
        self, merging_segment_list: List[MergedSegment]
    ) -> EntityPair:
        """merged segment のみをセグメントとするentity_pairを作成"""
        entity_l_new = copy.deepcopy(self.entity_l)
        entity_r_new = copy.deepcopy(self.entity_r)
        merged_segment_new = copy.deepcopy(merging_segment_list)
        entity_pair_new = EntityPair(entity_l_new, entity_r_new)
        # 既存のセグメントをすべて破棄し、merging_segment_listのみを採用する
        entity_pair_new.merged_segment_list = merged_segment_new
        # 並べ替える
        entity_pair_new.merged_segment_list = (
            entity_pair_new._sort_merged_segment_list()
        )
        return entity_pair_new


    def _sort_merged_segment_list(self) -> List[MergedSegment]:
        """並べ替える（左のみ、右のみ、両方）"""
        tmp = []
        tmp.extend(
            sorted(
                filter(
                    lambda x: len(x.segment_list_in_r) == 0
                    and len(x.segment_list_in_l) != 0,
                    self.merged_segment_list,
                ),
                key=lambda x: x.segment_list_in_l[0],
            )
        )
        tmp.extend(
            sorted(
                filter(
                    lambda x: len(x.segment_list_in_l) == 0
                    and len(x.segment_list_in_r) != 0,
                    self.merged_segment_list,
                ),
                key=lambda x: x.segment_list_in_r[0],
            )
        )
        tmp.extend(
            sorted(
                filter(
                    lambda x: len(x.segment_list_in_l) != 0
                    and len(x.segment_list_in_r) != 0,
                    self.merged_segment_list,
                ),
                key=lambda x: (x.segment_list_in_l[0], x.segment_list_in_r[0]),
            )
        )
        return tmp
