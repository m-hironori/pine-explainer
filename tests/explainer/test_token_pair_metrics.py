from pine.explainer.token_pair_metrics import (
    determine_word_relationship,
    calculate_cosine_similarities_with_mean_pooling,
)


def test_calculate_cosine_similarities_with_mean_pooling():
    # 類似度の計算
    word_pairs = [
        ("dog", "dog"),
        ("cat", "cat"),
        ("dog", "cat"),
        ("cat", "dog"),
    ]
    similarities = calculate_cosine_similarities_with_mean_pooling(word_pairs)

    assert len(similarities) == 4
    # 同じ単語ペアの類似度は、小数点第６位を四捨五入したら1.0になる
    assert round(similarities[0], 6) == 1.0
    assert round(similarities[1], 6) == 1.0
    # 単語ペアを入れ替えても値は同じ
    assert round(similarities[2], 6) == round(similarities[3], 6)
    # 同じ単語ペアのほうが類似度が大きい
    assert similarities[0] > similarities[2]


def test_determine_word_relationship():
    word1 = "apple"
    word2 = "orange"
    for hypernym_depth in range(1, 3):
        relationship = determine_word_relationship(word1, word2, hypernym_depth)
        if relationship == "same_category":
            break
    assert relationship == "same_category"

    word1 = "good"
    word2 = "bad"
    relationship = determine_word_relationship(word1, word2)
    assert relationship == "antonym"

    word1 = "apple"
    word2 = "good"
    relationship = determine_word_relationship(word1, word2)
    assert relationship is None

    word1 = "Apple_Computer"
    word2 = "Microsoft"
    relationship = determine_word_relationship(word1, word2)
    assert relationship is None
