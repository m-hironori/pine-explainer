# Magellan 並列実行
set -x
PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=0 GPU_ID=1 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=1 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=2 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=3 GPU_ID=1 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=4 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=5 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=6 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=7 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=8 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=9 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=10 GPU_ID=1 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/wym:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=11 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p GPU_ID ${GPU_ID} \
              make_wym_model.ipynb make_wym_model_${TARGET_DATASET_ID}.ipynb \
              >make_wym_model_${TARGET_DATASET_ID}.log 2>&1 &'

set +x
