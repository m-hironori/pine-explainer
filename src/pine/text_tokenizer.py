from typing import List, Tuple
from functools import lru_cache
import re
from dataclasses import dataclass

@dataclass
class TokenPos:
    """文字位置データ"""
    start: int
    end: int


def regex_tokenizer(text: str, sep_regex: str = "\s+") -> Tuple[List[str], List[TokenPos]]:
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


@lru_cache(maxsize=1)
def _get_spacy_nlp(model: str = "en_core_web_sm"):
    """Load spaCy pipeline once (cached) to avoid repeated spacy.load() overhead."""
    import spacy

    return spacy.load(model)


def phrase_tokenizer(
    text: str,
    max_tokens: int = 6,  # prevent overly long phrases (try 8 for product titles)
    min_tokens: int = 2,  # drop too-short phrase candidates
) -> Tuple[List[str], List[TokenPos]]:
    """Extract non-overlapping phrases (dependency-subtree) and also output all leftover
    tokens so that concatenating outputs reconstructs the original text.

    Returns:
      phrases: List[str]        # phrases and leftover pieces (incl. whitespace)
      phrase_poss: List[TokenPos]   # (start_char, end_char) for each element

    Reconstruction guarantee:
      "".join(phrases) == text

    Notes:
      - Uses spaCy dependency parse. Requires: spacy + en_core_web_sm.
      - The output contains:
          * selected phrase spans (non-overlapping)
          * everything else split into small pieces (whitespace and non-whitespace tokens)
    """
    nlp = _get_spacy_nlp("en_core_web_sm")
    doc = nlp(text)

    # --- 1) candidate generation (dep-subtree expansion) ---
    include_deps = {"compound", "amod", "nummod", "appos", "poss", "npadvmod"}
    split_punct = {",", ";", "|", "•", "."}
    split_words = {
        "under",
        "to",
        "in",
        "with",
        "for",
        "by",
        "from",
        "on",
        "at",
        "into",
        "over",
        "and",
    }

    def expand_root(root):
        idxs = {root.i}
        stack = [root]
        seen = {root.i}
        while stack:
            t = stack.pop()
            for c in t.children:
                if c.i in seen:
                    continue
                if c.dep_ in include_deps:
                    idxs.add(c.i)
                    seen.add(c.i)
                    stack.append(c)
                elif c.dep_ == "prep":
                    # include PP subtree (can make it long; will be trimmed later)
                    for g in c.subtree:
                        idxs.add(g.i)
                        seen.add(g.i)

        lo, hi = min(idxs), max(idxs)
        return (lo, hi, root.i)  # token span [lo, hi] inclusive + root index

    candidates = []
    for ch in doc.noun_chunks:
        candidates.append(expand_root(ch.root))

    # extra anchors for short / weird titles
    for t in doc:
        if t.pos_ in {"NOUN", "PROPN"} and t.dep_ in {
            "ROOT",
            "pobj",
            "dobj",
            "attr",
            "conj",
            "appos",
        }:
            candidates.append(expand_root(t))

    # --- 2) trimming: cut too-long spans nicely, keeping root inside ---
    def best_subspan_containing_root(lo: int, hi: int, root_i: int) -> Tuple[int, int]:
        # token indices [lo, hi] inclusive
        toks = list(range(lo, hi + 1))

        # cut points in token-index space (segment boundaries)
        cut_points = {lo, hi + 1}
        for i in toks:
            tok = doc[i]
            if tok.text in split_punct:
                cut_points.add(i)
                cut_points.add(i + 1)
            if tok.lower_ in split_words:
                # soft boundary: prefer cutting BEFORE this word
                cut_points.add(i)

        cut_points = sorted(p for p in cut_points if lo <= p <= hi + 1)

        # enumerate segments [a,b) that contain root_i
        segments = []
        for a in cut_points:
            for b in cut_points:
                if a >= b:
                    continue
                if a < lo or b > hi + 1:
                    continue
                if not (a <= root_i < b):
                    continue
                length = b - a
                if length < min_tokens:
                    continue
                segments.append((a, b))

        if not segments:
            return (root_i, root_i + 1)

        def score(seg):
            a, b = seg
            length = b - a
            over = max(0, length - max_tokens)
            dist_left = root_i - a
            # penalize oversize strongly; prefer longer up to max_tokens; prefer cutting leading context
            return (-(over * 10), length, -dist_left)

        a, b = max(segments, key=score)

        if (b - a) > max_tokens:
            # force a window around root
            a = max(a, root_i - 2)
            b = min(b, a + max_tokens)
            if not (a <= root_i < b):
                a = max(lo, root_i - max_tokens // 2)
                b = min(hi + 1, a + max_tokens)

        return (a, b)

    # convert candidate token spans -> trimmed char spans
    meta = []  # (char_span, (tok_len, has_num, has_code))
    for lo, hi, root_i in candidates:
        a, b = best_subspan_containing_root(lo, hi, root_i)
        span = doc[a:b]  # Span

        start = span.start_char
        end = span.end_char

        # trim whitespace
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

        if end - start < 3:
            continue

        span_text = text[start:end]
        has_num = 1 if re.search(r"\d", span_text) else 0
        has_code = 1 if re.search(r"[A-Za-z]\d|\d[A-Za-z]", span_text) else 0
        tok_len = len(doc[a:b])
        meta.append(((start, end), (tok_len, has_num, has_code)))

    # --- 3) remove duplicates (keep better score) ---
    uniq = {}
    for sp, sc in meta:
        if sp not in uniq:
            uniq[sp] = sc
        else:
            uniq[sp] = max(uniq[sp], sc)

    spans = list(uniq.keys())

    # --- 4) select non-overlapping spans (no nesting/overlap) ---
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))

    def span_score(sp: TokenPos):
        tok_len, has_num, has_code = uniq[sp]
        length_chars = sp[1] - sp[0]
        return (min(tok_len, max_tokens), has_code, has_num, length_chars)

    selected: List[TokenPos] = []
    i = 0
    while i < len(spans_sorted):
        cur = spans_sorted[i]
        group = [cur]
        j = i + 1
        union_end = cur[1]

        while j < len(spans_sorted) and spans_sorted[j][0] < union_end:
            group.append(spans_sorted[j])
            union_end = max(union_end, spans_sorted[j][1])
            j += 1

        best = max(group, key=span_score)
        selected.append(best)
        i = j

    # --- 5) build output WITHOUT whitespace-only elements ---
    # We keep each selected phrase span as ONE piece.
    # For gaps, we output only non-whitespace tokens (\S+).
    selected = sorted(selected, key=lambda x: x[0])

    def split_gap(gap_text: str, gap_start: int) -> Tuple[List[str], List[TokenPos]]:
        pieces: List[str] = []
        poss: List[TokenPos] = []
        for m in re.finditer(r"\S+", gap_text):
            a = gap_start + m.start()
            b = gap_start + m.end()
            pieces.append(text[a:b])
            poss.append(TokenPos(a, b))
        return pieces, poss

    phrases: List[str] = []
    phrase_poss: List[TokenPos] = []

    cursor = 0
    for (a, b) in selected:
        if cursor < a:
            gap_p, gap_s = split_gap(text[cursor:a], cursor)
            phrases.extend(gap_p)
            phrase_poss.extend(gap_s)

        phrases.append(text[a:b])
        phrase_poss.append(TokenPos(a, b))
        cursor = b

    if cursor < len(text):
        gap_p, gap_s = split_gap(text[cursor:], cursor)
        phrases.extend(gap_p)
        phrase_poss.extend(gap_s)

    return phrases, phrase_poss


