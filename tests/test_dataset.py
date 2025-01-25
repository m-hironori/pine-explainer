import tempfile

from pine.dataset import load_dataset


# def test_load_dataset():
#     dataset_names = [
#         "structured_amazon_google",
#         "structured_beer",
#         "structured_dblp_acm",
#         "structured_dblp_google_scholar",
#         "structured_fodors_zagat",
#         "structured_walmart_amazon",
#         "structured_itunes_amazon",
#         "dirty_dblp_acm",
#         "dirty_dblp_google_scholar",
#         "dirty_walmart_amazon",
#         "dirty_itunes_amazon",
#         "textual_abt_buy",
#         "textual_company",
#     ]
#     tmp_dir = tempfile.TemporaryDirectory()
#     for dataset_name in dataset_names:
#         dataset = load_dataset(dataset_name, tmp_dir.name)
#         for kind in ["train", "val", "test"]:
#             assert hasattr(dataset, kind), f"{kind}が{dataset_name}に存在しない"
#             for recordsprop in ["records", "record_id_pairs", "labels"]:
#                 dataset_kind = getattr(dataset, kind)
#                 assert hasattr(
#                     dataset_kind, recordsprop
#                 ), f"{recordsprop}が{dataset_name}.{kind}に存在しない"
