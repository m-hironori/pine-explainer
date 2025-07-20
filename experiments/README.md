# Experimental Code Folder

## Execution Procedure

### Create Matcher Models

- Create DITTO Model
  - Run `01_make_model/make_DITTO_model_by_lemon.ipynb`
- Create py_entitymatching Model
  - Run `01_make_model/make_py_entitymatching_model_by_lemon.ipynb`
- Compare Model Accuracy
  - Run `01_make_model/compare_model_evaluation.ipynb`

### Create LIME and Lemon Explanations

- Create LIME Explanations
  - Run `cd 02_make_explanation; bash make_lime_results_ditto.sh`
  - Run `cd 02_make_explanation; bash make_lime_results_magellan.sh`
- Create LEMON Explanations
  - Run `cd 02_make_explanation; bash make_lemon_results_ditto.sh`
  - Run `cd 02_make_explanation; bash make_lemon_results_magellan.sh`

### Create PINE, CosSim, LIME_pair, LIME_r, and Lemon_r Explanations

- Create PINE Explanations
  - Run `cd 02_make_explanation; bash eval_pine.sh`
- Create CosSim Explanations
  - Run `cd 02_make_explanation; bash eval_cossim.sh`
- Create LIME_pair Explanations
  - Run `cd 02_make_explanation; bash eval_lime_pair.sh`
- Create LIME_r Explanations
  - Run `cd 02_make_explanation; bash eval_lime_ranked.sh`
- Create LEMON_r Explanations
  - Run `cd 02_make_explanation; bash eval_lemon_ranked.sh`

### Comparative Evaluation with Metrics

Run `03_evaluations/compare_methods_revised.ipynb`

### Comparative Evaluation of WYM

- Create WYM Model
  - Run `cd 04_evaluations_wym; bash make_wym_model.sh`
- Create WYM Explanations
  - Run `cd 04_evaluations_wym; bash eval_wym.sh`
- Create PINE Explanations
  - Run `cd 04_evaluations_wym; bash make_lime_results_wym_matcher.sh`
  - Run `cd 04_evaluations_wym; bash make_wym_pine.sh`
- Comparative Evaluation with Metrics
  - Run `04_evaluations/compare_methods_wym_revised.ipynb`