PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=0 TOP_N=5 GPU_ID=0 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=1 TOP_N=5 GPU_ID=0 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=2 TOP_N=5 GPU_ID=0 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=3 TOP_N=5 GPU_ID=0 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=4 TOP_N=5 GPU_ID=3 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=5 TOP_N=5 GPU_ID=1 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=6 TOP_N=5 GPU_ID=2 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=7 TOP_N=5 GPU_ID=2 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=8 TOP_N=5 GPU_ID=2 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=9 TOP_N=5 GPU_ID=3 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=10 TOP_N=5 GPU_ID=3 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null

PYTHONPATH=/workspaces/entity-matching-explainer/lemon/src:/workspaces/entity-matching-explainer \
    TARGET_DATASET_ID=11 TOP_N=5 GPU_ID=3 \
    nohup bash -c 'papermill  \
              --prepare-execute --log-output \
              -p TARGET_DATASET_ID ${TARGET_DATASET_ID} -p TOP_N ${TOP_N} -p GPU_ID ${GPU_ID}\
              eval_lime_ranked.ipynb eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.ipynb \
              >eval_lime_ranked_${TARGET_DATASET_ID}_${TOP_N}.log 2>&1 &' >/dev/null
