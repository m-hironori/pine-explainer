# Magellan 並列実行
set -x
PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=0 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=2400 GPU_ID=0\
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=1 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=100 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=2 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=2500 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=3 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=5800 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=4 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=200 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=5 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=2100 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=6 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=200 GPU_ID=1 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=7 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=2500 GPU_ID=1 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=8 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=5800 GPU_ID=1 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=9 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=2100 GPU_ID=3 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=10 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=200 GPU_ID=3 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'


PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=11 TOP_N=5 TARGET_MATCHER_ID=1 START_DATA_IDX=0 END_DATA_IDX=2000 GPU_ID=0 \
    bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} \
              -p TARGET_MATCHER_ID ${TARGET_MATCHER_ID} \
              -p GPU_ID ${GPU_ID} \
              -p START_DATA_IDX ${START_DATA_IDX} -p END_DATA_IDX ${END_DATA_IDX} \
              make_lemon_results.ipynb make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.ipynb \
              >make_lemon_results_${TARGET_DATASET_ID}_${TOP_N}_${TARGET_MATCHER_ID}_${START_DATA_IDX}_${END_DATA_IDX}.log 2>&1 &'

set +x
