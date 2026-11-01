
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TYPR="${1:-bye}"
# ========= 配置区 =========
FIX_CUDA_DEVICES="0" # 固定cuda，所有任务共用这个
# 任务列表：每行 TRIGGER DATASET LORA_ADAPTER_PATH MODEL
task_list=(
# "badnet flickr30k llava"
# "badnet iconqa llava"
# "badnet iconqa internvl"
# "badnet scienceqa llava"
# "badnet scienceqa internvl"
# "badnet vqa llava"

# "blended flickr30k llava"
# "blended iconqa llava"
# "blended iconqa internvl"
# "blended scienceqa llava"
# "blended scienceqa internvl"
# "blended vqa llava"

# "sig iconqa llava"
# "sig iconqa internvl"
# "sig scienceqa llava"

# "wanet flickr30k llava"
# "wanet iconqa llava"
# "wanet iconqa internvl"
# "wanet scienceqa llava"
# "wanet scienceqa internvl"

# "ftrojan flickr30k llava"
# "ftrojan iconqa llava"
# "ftrojan scienceqa llava"
# "ftrojan vqa llava"

# "badnet flickr30k internvl"
# "badnet vqa internvl "
# "blended flickr30k internvl"
# "blended vqa internvl "
# "sig flickr30k llava "
# "sig flickr30k internvl "
# "sig scienceqa internvl "
# "sig vqa llava "
# "sig vqa internvl "
# "wanet flickr30k internvl "
# "wanet vqa llava "
# "wanet vqa internvl "
# "ftrojan flickr30k internvl"
# "ftrojan iconqa internvl "
# "ftrojan scienceqa internvl"
# "ftrojan vqa internvl "
# "badnet iconqa qwen"
"blended iconqa qwen"
)
# ==========================

for task in "${task_list[@]}"; do
  read -r TRIGGER DATASET MODEL <<< "${task}"
  echo "====================================================="
  echo "TRIGGER=${TRIGGER}"
  echo "DATASET=${DATASET}"
  echo "MODEL=${MODEL}"
  echo "====================================================="

  # RAW_BASE_DIR：原始raw_index.jsonl所在目录（不带tcap_bye后缀）
  LABLE_PATH="/data1/shizhili/backdoor_dataset/data_v2/data_${TRIGGER}/${DATASET}/exp3_and/train.jsonl"

  TCAP_FEATURES_PATH="/data1/shizhili/two_stage_v4_only_atten_one_tcap_bye/${TRIGGER}/${DATASET}/${MODEL}/${TYPR}_features.jsonl"
  
  SCORES_JSONL_PATH="/home/pbxs5/shizhili/code/two_stage_fixed_v1_4_finally/scores/${TYPR}/${TRIGGER}_${DATASET}_${MODEL}_scores.jsonl"
  SCORES_NPZ_PATH="/home/pbxs5/shizhili/code/two_stage_fixed_v1_4_finally/scores/${TYPR}/${TRIGGER}_${DATASET}_${MODEL}_scores.npz"
 
  echo "LABLE_PATH=${LABLE_PATH}"
  echo "TCAP_FEATURES_PATH=${TCAP_FEATURES_PATH}"
  echo "SCORES_JSONL_PATH=${SCORES_JSONL_PATH}"
  echo "SCORES_NPZ_PATH=${SCORES_NPZ_PATH}"
  echo "====================================================="
  
  python eval_tcap_bye.py \
  --features "${TCAP_FEATURES_PATH}"  \
  --labels "${LABLE_PATH}" \
  --save "${SCORES_JSONL_PATH}" \
  --save-scores "${SCORES_NPZ_PATH}"


  
  echo "✅ 该任务执行完成"
done
echo -e "\n🎉 全部任务串行跑完！"



python eval.py \
--features "./gen_features/features/blended_scienceqa_internvl/tcap_features.jsonl"  \
--labels "/home/pbxs5/shizhili/JACE_Project_test/dataset/data_blended_t1_token/scienceqa/exp3_and/train.jsonl" \
--save "./scores/blended_scienceqa_internvl.josn" \
--save-scores "./scores/blended_scienceqa_internvl.npz"