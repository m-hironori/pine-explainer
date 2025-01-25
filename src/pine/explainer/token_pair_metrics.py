import nltk
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

nltk.download("wordnet")

device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルとトークナイザーの準備
bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)


def calculate_cosine_similarities_with_mean_pooling(word_pairs):
    # 0件なら0件で返す
    if len(word_pairs) == 0:
        return np.array([])

    word1_list = [pair[0] for pair in word_pairs]
    word2_list = [pair[1] for pair in word_pairs]

    inputs1 = bert_tokenizer(
        word1_list, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(bert_model.device)
    inputs2 = bert_tokenizer(
        word2_list, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(bert_model.device)

    with torch.no_grad():
        outputs1 = bert_model(**inputs1)
        outputs2 = bert_model(**inputs2)

    # 平均Poolingを行う
    mask1 = (
        inputs1["attention_mask"]
        .unsqueeze(-1)
        .expand_as(outputs1.last_hidden_state)
        .float()
    )
    mask2 = (
        inputs2["attention_mask"]
        .unsqueeze(-1)
        .expand_as(outputs2.last_hidden_state)
        .float()
    )

    embed1 = torch.sum(outputs1.last_hidden_state * mask1, 1) / mask1.sum(1)
    embed2 = torch.sum(outputs2.last_hidden_state * mask2, 1) / mask2.sum(1)

    # コサイン類似度を計算し、numpy配列として返す
    similarities = (
        torch.nn.functional.cosine_similarity(embed1, embed2).to("cpu").numpy()
    )
    return similarities

def get_hypernyms_recursive(synset, depth=2):
    """指定した深さまで上位語を再帰的に取得する"""
    hypernyms = set()
    if depth > 0:
        direct_hypernyms = synset.hypernyms()
        hypernyms.update(direct_hypernyms)
        for hypernym in direct_hypernyms:
            hypernyms.update(get_hypernyms_recursive(hypernym, depth - 1))
    return hypernyms


def determine_word_relationship(word1: str, word2: str, hypernym_depth: int = 2) -> str:
    # 同義語、対義語、同カテゴリのフラグを初期化
    is_synonym = False
    is_antonym = False
    is_same_category = False

    # word1とword2のシンセットを取得
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    # 同義語かどうかを判別
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1 == synset2:
                is_synonym = True
                break
        if is_synonym:
            break

    # 対義語かどうかを判別
    for synset1 in synsets1:
        for lemma in synset1.lemmas():
            if is_antonym:
                break
            for antonym in lemma.antonyms():
                if antonym.name() == word2:
                    is_antonym = True
                    break

    # 同じカテゴリの単語かどうかを判別
    for synset1 in synsets1:
        for synset2 in synsets2:
            # 各シンセットの上位語を深さ3まで取得
            hypernyms1 = get_hypernyms_recursive(synset1, depth=hypernym_depth)
            hypernyms2 = get_hypernyms_recursive(synset2, depth=hypernym_depth)
            # 共通の上位語が存在するか確認
            common_hypernyms = hypernyms1.intersection(hypernyms2)
            if common_hypernyms:
                is_same_category = True
                break
        if is_same_category:
            break

    # 判定結果を返す
    if is_synonym:
        return "synonym"
    elif is_antonym:
        return "antonym"
    elif is_same_category:
        return "same_category"
    else:
        return None