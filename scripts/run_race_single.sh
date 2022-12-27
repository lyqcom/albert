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
echo "bash scripts/run_race_single.sh"
echo "for example: bash scripts/run_race_single.sh"
echo "=============================================================================================================="

mkdir -p ../output/logs/
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/output/logs/
export GLOG_logtostderr=0
python ${PROJECT_DIR}/../run_race.py  \
    --config_path="../task_race_config.yaml" \
    --device_target="GPU" \
    --do_train=True \
    --do_eval=True \
    --distribute=False \
    --epoch_num=15000 \
    --max_qa_length=128 \
    --train_data_shuffle=True \
    --eval_data_shuffle=False \
    --train_batch_size=6 \
    --eval_batch_size=4 \
    --save_checkpoints_steps=100 \
    --warmup_step=1000 \
    --save_finetune_checkpoint_path="./output/race_finetune/" \
    --load_last_finetune_checkpoint_path="" \
    --load_pretrain_checkpoint_path="./moveckpt/albert_base.ckpt" \
    --load_finetune_checkpoint_path="./output/race_finetune/" \
    --eval_dir_path="./data/finetune/" \
    --train_dir_path="./data/finetune/" \
    --train_mindrecord_file="./data/finetune/mindrecord/race/train384.mindrecord" \
    --eval_mindrecord_file="./data/finetune/mindrecord/race/eval384.mindrecord" \
    --vocab_file_path="./moveckpt/30k-clean.vocab" > ./output/logs/race_log.log 2>&1 &
