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
echo "for example: bash run_pretrain_single.sh"
echo "It is better to use absolute path."
echo "=============================================================================================================="
mkdir -p ../output/logs/
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/output/logs/
export GLOG_logtostderr=0

python ${PROJECT_DIR}/../run_pretrain.py  \
    --config_path="../pretrain_config.yaml" \
    --device_target="GPU"      \
    --distribute=True        \
    --epoch=10    \
    --device_num=2    \
    --train_batch_size=96    \
    --eval_batch_size=64    \
    --enable_save_ckpt=True        \
    --enable_lossscale=True    \
    --do_shuffle=True        \
    --enable_data_sink=True        \
    --data_sink_steps=20        \
    --accumulation_steps=1        \
    --save_checkpoint_path='./model/en_test'        \
    --load_checkpoint_path=""      \
    --save_checkpoint_steps=5000  \
    --train_steps=5000  \
    --save_checkpoint_num=1      \
    --data_dir='./data/pre_train/train_data/run'  > ./output/logs/pre_train_s.log 2>&1 &

