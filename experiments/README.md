# 実験用コードフォルダ

## 実行手順

### Matcherモデルを作成

- DITTOモデルの作成
  - `01_make_model/make_DITTO_model_by_lemon.ipynb` を実行
- py_entitymacching モデルの作成
  - `01_make_model/make_py_entitymatching_model_by_lemon.ipynb` を実行
- モデルの精度比較
  - `01_make_modelcompare_model_evaluation.ipynb` を実行

### LIME, Lemon の explanation を作成

- LIME のexplanation作成
  - `cd 02_make_explanation; bash make_lime_results_ditto.sh` を実行
  - `cd 02_make_explanation; bash make_lime_results_magellan.sh` を実行
- LEMON のexplanationの作成
  - `cd 02_make_explanation; bash make_lemon_results_ditto.sh` を実行
  - `cd 02_make_explanation; bash make_lemon_results_magellan.sh` を実行

### PINE, CosSim, LIME_pair, LIME_r, Lemon_r の explanation を作成

- PINE の explanation作成
  - `cd 02_make_explanation; bash eval_pine.sh` を実行
- CosSim の explanation作成
  - `cd 02_make_explanation; bash eval_cossim.sh` を実行
- LIME_pair の explanation作成
  - `cd 02_make_explanation; bash eval_lime_pair.sh` を実行
- LIME_r の explanation作成
  - `cd 02_make_explanation; bash eval_lime_ranked.sh` を実行
- LEMON_r の explanation作成
  - `cd 02_make_explanation; bash eval_lemon_ranked.sh` を実行

### Metrics で比較評価

`03_evaluations/compare_methods_revised.ipynb` を実行

### WYM を比較評価

- WYM モデル作成
  - `cd 04_evaluations_wym; bash make_wym_model.sh` を実行
- WYM の explanation の作成
  - `cd 04_evaluations_wym; bash eval_wym.sh` を実行
- PINE の explanation の作成
  - `cd 04_evaluations_wym; bash make_lime_results_wym_matcher.sh`
  - `cd 04_evaluations_wym; bash make_wym_pine.sh`
- Metricsで比較評価
  - `04_evaluations/compare_methods_wym_revised.ipynb` を実行
