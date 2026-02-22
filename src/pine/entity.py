from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple

import pandas as pd

from .text_tokenizer import TokenPos, regex_tokenizer


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
    """Class representing an Entity with attributes and segments."""

    def __init__(
        self,
        attr_list: List[Attribute] = [],
        tokenizer_func: Callable = regex_tokenizer,
        aggregate_same_word: bool = True,
        segment_list: Optional[List[List[SegmentPart]]] = None,
    ) -> None:
        self.attr_list: List[Attribute] = attr_list
        self.tokenizer_func = tokenizer_func
        self.aggregate_same_word = aggregate_same_word

        if segment_list is not None:
            self.segment_list = segment_list
        else:
            self.segment_list = self._make_segments(
                attr_list, tokenizer_func, aggregate_same_word
            )

    def _make_segments(
        self,
        attr_list: List[Attribute],
        tokenizer_func: Callable,
        aggregate_same_word: bool,
    ) -> List[List[SegmentPart]]:
        """Creates segments from attributes."""
        segment_list: List[List[SegmentPart]] = []
        word_to_seg: Dict[str, List[SegmentPart]] = {}
        for attr_index, attr in enumerate(attr_list):
            if attr.dtype == "string":
                words, poslist = make_word_poslist(
                    attr.value, tokenizer_func, aggregate_same_word
                )
                for word, poss in zip(words, poslist):
                    segment = [
                        SegmentPart(attr_index, pos.start, pos.end) for pos in poss
                    ]
                    if aggregate_same_word and word in word_to_seg:
                        word_to_seg[word].extend(segment)
                    else:
                        word_to_seg[word] = segment
                        segment_list.append(segment)
            else:
                if attr.value is None:
                    continue
                segment = [SegmentPart(attr_index, None, None)]
                segment_list.append(segment)
        return segment_list

    def segment_size(self) -> int:
        return len(self.segment_list)

    def get_segment_label(self, index: int) -> str:
        segment_parts = self.segment_list[index]
        attr = self.attr_list[segment_parts[0].attr_index]
        if attr.dtype == "string":
            return attr.value[segment_parts[0].start : segment_parts[0].end]
        if attr.value is None:
            return None
        return str(attr.value)

    def get_attribute_list_by_segments(self, index_list: List[int]) -> List[Attribute]:
        value_seg_parts: Dict[int, List[SegmentPart]] = {}
        for seg_idx in index_list:
            for seg_part in self.segment_list[seg_idx]:
                if seg_part.attr_index not in value_seg_parts:
                    value_seg_parts[seg_part.attr_index] = []
                value_seg_parts[seg_part.attr_index].append(seg_part)

        ret_attr_list = []
        sorted_attr_indices = sorted(value_seg_parts.keys())

        for target_attr_idx in sorted_attr_indices:
            target_segparts = value_seg_parts[target_attr_idx]
            attr_org = self.attr_list[target_attr_idx]

            if attr_org.dtype != "string":
                attr = Attribute(attr_org.name, attr_org.value, attr_org.dtype)
                ret_attr_list.append(attr)
                continue

            val = ""
            target_segparts = sorted(target_segparts, key=lambda x: x.start)
            for target_segpart in target_segparts:
                val += " " + attr_org.value[target_segpart.start : target_segpart.end]

            val = re.sub(r"\s+", " ", val).strip()
            attr = Attribute(attr_org.name, val, attr_org.dtype)
            ret_attr_list.append(attr)
        return ret_attr_list

    def make_entity_by_deleting_segments(
        self,
        delete_segments_id_list: List[int],
        mask_str: str = None,
        exclude_non_segment_chars: bool = False,
    ) -> "Entity":
        """
        指定されたセグメントを削除（またはマスク）します。

        Arguments:
            delete_segments_id_list: 削除/マスク対象のセグメントIDリスト
            mask_str: 削除部分を置き換える文字列
            exclude_non_segment_chars: Trueの場合、どのセグメントにも属さない文字（記号等）を空白に置き換える
        """
        new_attr_list = []
        attr_index_maps: Dict[int, List[int]] = {}

        for i, attr in enumerate(self.attr_list):
            if attr.dtype == "string":
                original_val = attr.value
                val_len = len(original_val)

                is_in_deleted_segment = [False] * val_len
                is_in_any_segment = [False] * val_len

                for s_idx, seg in enumerate(self.segment_list):
                    for p in seg:
                        if p.attr_index == i:
                            for char_idx in range(p.start, p.end):
                                is_in_any_segment[char_idx] = True
                                if s_idx in delete_segments_id_list:
                                    is_in_deleted_segment[char_idx] = True

                temp_chars = []
                old_to_temp_idx = [0] * (val_len + 1)
                current_temp_pos = 0
                prev_was_space = False
                in_deleted_span = False

                for char_idx in range(val_len):
                    old_to_temp_idx[char_idx] = current_temp_pos

                    is_deleted = is_in_deleted_segment[char_idx]
                    is_noise = not is_in_any_segment[char_idx]

                    if is_deleted:
                        if mask_str is not None and not in_deleted_span:
                            for m_char in mask_str:
                                temp_chars.append(m_char)
                                current_temp_pos += 1
                            prev_was_space = (
                                mask_str[-1].isspace() if mask_str else prev_was_space
                            )
                            in_deleted_span = True
                        # mask_strがNoneの場合は削除される
                    elif exclude_non_segment_chars and is_noise:
                        # セグメント外文字を削除ではなく空白に置き換え
                        in_deleted_span = False
                        if not prev_was_space:
                            temp_chars.append(" ")
                            current_temp_pos += 1
                            prev_was_space = True
                    else:
                        # 保持対象（既存セグメント内、または exclude_non_segment_chars=False 時のノイズ）
                        in_deleted_span = False
                        char = original_val[char_idx]
                        is_space = char.isspace()

                        if is_space:
                            if not prev_was_space:
                                temp_chars.append(char)
                                current_temp_pos += 1
                                prev_was_space = True
                        else:
                            temp_chars.append(char)
                            current_temp_pos += 1
                            prev_was_space = False

                old_to_temp_idx[val_len] = current_temp_pos

                temp_str = "".join(temp_chars)
                trimmed_str = temp_str.strip()

                if not trimmed_str:
                    attr_index_maps[i] = [0] * (val_len + 1)
                    new_attr_list.append(Attribute(attr.name, "", attr.dtype))
                    continue

                start_offset = len(temp_str) - len(temp_str.lstrip())
                end_limit = start_offset + len(trimmed_str)

                final_mapping = [0] * (val_len + 1)
                for old_idx in range(val_len + 1):
                    t_pos = old_to_temp_idx[old_idx]
                    if t_pos <= start_offset:
                        final_mapping[old_idx] = 0
                    elif t_pos >= end_limit:
                        final_mapping[old_idx] = len(trimmed_str)
                    else:
                        final_mapping[old_idx] = t_pos - start_offset

                attr_index_maps[i] = final_mapping
                new_attr_list.append(Attribute(attr.name, trimmed_str, attr.dtype))
            else:
                is_deleted = False
                for seg_idx in delete_segments_id_list:
                    if any(p.attr_index == i for p in self.segment_list[seg_idx]):
                        is_deleted = True
                        break

                val = attr.value
                new_dtype = attr.dtype
                if is_deleted:
                    val = mask_str
                    if mask_str is not None:
                        new_dtype = "string"
                new_attr_list.append(Attribute(attr.name, val, new_dtype))

        keep_indices = [
            idx
            for idx in range(self.segment_size())
            if idx not in delete_segments_id_list
        ]
        new_segment_list = []

        for idx in keep_indices:
            cloned_parts = []
            for p in self.segment_list[idx]:
                if self.attr_list[p.attr_index].dtype == "string":
                    mapping = attr_index_maps[p.attr_index]
                    new_start = mapping[p.start]
                    new_end = mapping[p.end]
                    cloned_parts.append(SegmentPart(p.attr_index, new_start, new_end))
                else:
                    cloned_parts.append(SegmentPart(p.attr_index, p.start, p.end))
            new_segment_list.append(cloned_parts)

        return Entity(
            attr_list=new_attr_list,
            tokenizer_func=self.tokenizer_func,
            aggregate_same_word=self.aggregate_same_word,
            segment_list=new_segment_list,
        )

    def make_entity_by_remain_segments(
        self,
        remain_segments_id_list: List[int],
        mask_str: str = None,
        exclude_non_segment_chars: bool = True,
    ) -> "Entity":
        """
        指定されたセグメントのみを保持します。
        exclude_non_segment_chars が True の場合、セグメント間の記号等は空白に変換されます。
        """
        all_indices = set(range(self.segment_size()))
        remain_set = set(remain_segments_id_list)
        delete_indices = list(all_indices - remain_set)

        return self.make_entity_by_deleting_segments(
            delete_indices,
            mask_str=mask_str,
            exclude_non_segment_chars=exclude_non_segment_chars,
        )

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
        return Entity(org_attr_list, self.tokenizer_func, self.aggregate_same_word)

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
        # df = pd.DataFrame(
        #     data={attr.name: [attr.value] for attr in self.attr_list},
        #     columns=[attr.name for attr in self.attr_list],
        # )
        # df.index.name = "__id"
        # for col, attr in zip(df.columns, self.attr_list):
        #     df[col] = df[col].astype(attr.dtype)
        df = pd.DataFrame(
            {
                attr.name: pd.Series([attr.value], dtype=attr.dtype)
                for attr in self.attr_list
            },
        )
        df.index.name = "__id"
        return df

    def from_dataframe(
        df: pd.DataFrame,
        tokenizer_func: Callable = regex_tokenizer,
        aggregate_same_word: bool = True,
    ) -> Entity:
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
        return Entity(attr_list, tokenizer_func, aggregate_same_word)


class MergedSegment(NamedTuple):
    segment_list_in_l: List[int]
    segment_list_in_r: List[int]


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
            merged_segment_new = MergedSegment([], [])
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
        self, merging_segment_list: List[MergedSegment], is_sorted: bool = True
    ) -> EntityPair:
        """merged segment のみをセグメントとするentity_pairを作成"""
        entity_l_new = copy.deepcopy(self.entity_l)
        entity_r_new = copy.deepcopy(self.entity_r)
        merged_segment_new = copy.deepcopy(merging_segment_list)
        entity_pair_new = EntityPair(entity_l_new, entity_r_new)
        # 既存のセグメントをすべて破棄し、merging_segment_listのみを採用する
        entity_pair_new.merged_segment_list = merged_segment_new
        # 並べ替える
        if is_sorted:
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
