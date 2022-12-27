#!/bin/bash
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_squad_v2_single.sh"
echo "for example: bash scripts/run_squad_v2_single.sh"
echo "=============================================================================================================="

mkdir -p ../output/logs/
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/output/logs/
export GLOG_logtostderr=0
python ${PROJECT_DIR}/../run_squad_v2.py  \
    --config_path="../task_squad_v2_config.yaml" \
    --device_target="GPU" \
    --do_train=True \
    --do_eval=True \
    --distribute=False \
    --epoch_num=3 \
    --train_batch_size=32 \
    --eval_batch_size=16 \
    --num_class=2 \
    --train_data_shuffle=True \
    --eval_data_shuffle=False \
    --vocab_file_path="./moveckpt/30k-clean.vocab" \
    --spm_model_file="./moveckpt/30k-clean.model" \
    --eval_json_path="./data/finetune/SQuAD1.1/dev-v1.1.json" \
    --train_json_path="./data/finetune/SQuAD1.1/train-v1.1.json" \
    --train_mindrecord_file="./data/finetune/mindrecord/squad_v2/train.mindrecord" \
    --eval_mindrecord_file="./data/finetune/mindrecord/squad_v2/eval.mindrecord" \
    --eval_pkl_file="./data/finetune/mindrecord/squad_v2/eval.pkl" \
    --save_finetune_checkpoint_path="./output/squad_v2_finetune/" \
    --load_last_finetune_checkpoint_path="" \
    --load_pretrain_checkpoint_path="./moveckpt/albert_base.ckpt" \
    --load_finetune_checkpoint_path="./output/squad_v2_finetune/"  > ./output/logs/squad_v2.log 2>&1 &
