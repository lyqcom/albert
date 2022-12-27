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

"""
Albert evaluation script.
"""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.metrics import Metric
from mindspore.ops import operations as P
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.albert_for_pre_training import AlbertPreTraining
from dataset import create_albert_dataset
from model_utils.config import get_config

cfg = get_config("../pretrain_config.yaml")
albert_net_cfg = cfg.albert_net_cfg


class myMetric(Metric):
    '''
    Self-defined Metric as a callback.
    '''

    def __init__(self):
        super(myMetric, self).__init__()
        self.clear()

    # 方法会把类中相关计算参数初始化。
    def clear(self):
        self.total_MLM_num = 0
        self.total_SOP_num = 0
        self.acc_MLM_num = 0
        self.acc_SOP_num = 0

    # 接受预测值和标签值，更新Accuracy内部变量。
    def update(self, *inputs):
        # 将数据类型转换为numpy数组
        total_MLM_num = self._convert_data(inputs[0])
        acc_MLM_num = self._convert_data(inputs[1])
        total_SOP_num = self._convert_data(inputs[2])
        acc_SOP_num = self._convert_data(inputs[3])
        self.total_MLM_num = total_MLM_num
        self.total_SOP_num = total_SOP_num
        self.acc_MLM_num = acc_MLM_num
        self.acc_SOP_num = acc_SOP_num

    # 方法会计算相关指标，返回计算结果。
    def eval(self):
        return self.acc_MLM_num / self.total_MLM_num, self.acc_SOP_num / self.total_SOP_num


class AlbertPretrainEva(nn.Cell):
    '''
    Evaluate MaskedLM prediction scores
    '''

    def __init__(self, config, is_training=False, use_one_hot_embeddings=False):
        super(AlbertPretrainEva, self).__init__()
        self.albert = AlbertPreTraining(config, is_training, use_one_hot_embeddings)
        self.argmax = P.Argmax(axis=-1, output_type=mstype.int32)
        self.equal = P.Equal()
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum()
        self.total_MLM = Parameter(Tensor([0], mstype.float32))
        self.total_SOP = Parameter(Tensor([0], mstype.float32))
        self.acc_MLM = Parameter(Tensor([0], mstype.float32))
        self.acc_SOP = Parameter(Tensor([0], mstype.float32))
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  token_boundary,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights):
        """Calculate prediction scores"""
        bs, _ = self.shape(input_ids)
        # 获取预测分数
        probs = self.albert(input_ids, input_mask, token_type_id, masked_lm_positions)
        index_MLM = self.argmax(probs[0])
        index_SOP = self.argmax(probs[1])
        index_MLM = self.reshape(index_MLM, (bs, -1))
        index_SOP = self.reshape(index_SOP, (bs, -1))
        # 比价两个tensor对应的值的位置是否相等
        eval_MLM_acc = self.equal(index_MLM, masked_lm_ids)
        eval_SOP_acc = self.equal(index_SOP, next_sentence_labels)
        eval_MLM_acc1 = self.cast(eval_MLM_acc, mstype.float32)
        eval_SOP_acc1 = self.cast(eval_SOP_acc, mstype.float32)
        real_acc = eval_MLM_acc1 * masked_lm_weights
        acc_MLM = self.sum(real_acc)
        acc_SOP = self.sum(eval_SOP_acc1)
        total_MLM = self.sum(masked_lm_weights)
        total_SOP = len(index_SOP)
        self.total_MLM += total_MLM
        self.total_SOP += total_SOP
        self.acc_MLM += acc_MLM
        self.acc_SOP += acc_SOP
        return self.total_MLM, self.acc_MLM, self.total_SOP, self.acc_SOP


def albert_predict():
    '''
    Predict function
    '''

    # 创建数据集
    dataset = create_albert_dataset(device_num=1, rank=0, do_shuffle="true", data_dir=cfg.data_file,
                                    batch_size=cfg.eval_batch_size)

    net_for_pretraining = AlbertPretrainEva(albert_net_cfg)
    net_for_pretraining.set_train(False)

    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net_for_pretraining, param_dict)

    model = Model(net_for_pretraining, eval_network=net_for_pretraining, eval_indexes=None,
                  metrics={'name': myMetric()})
    result = model.eval(dataset, dataset_sink_mode=False)
    print("==============================================================")
    for _, v in result.items():
        print("MLM Accuracy is: ", v[0])
        print("SOP Accuracy is: ", v[1])
    print("==============================================================")


if __name__ == "__main__":
    DEVICE_ID = 0
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=DEVICE_ID)
    context.set_context(reserve_class_name_in_scope=False)
    albert_predict()
