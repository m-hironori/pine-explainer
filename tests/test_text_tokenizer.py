import pytest
from pine.text_tokenizer import regex_tokenizer, TokenPos, phrase_tokenizer


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


def test_phrase_tokenizer():
    # 句のトークナイザーのテスト
    text = "epson 1500 hours 200w uhe projector lamp elplp12"
    expected_phrases = ["epson 1500 hours 200w uhe projector", "lamp", "elplp12"]
    expected_phrase_poss = [
        TokenPos(start=0, end=35),
        TokenPos(start=36, end=40),
        TokenPos(start=41, end=48),
    ]
    phrases, phrase_poss = phrase_tokenizer(text)
    assert phrases == expected_phrases
    assert phrase_poss == expected_phrase_poss